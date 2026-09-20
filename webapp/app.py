#!/usr/bin/env python3
"""
OLY VISION - Curation Web App
-----------------------------
Flask web application for curating VLM inference results with
multi-user support and batch-based task assignment.

Features:
- User authentication with database
- Role-based access (admin/curator)
- Batch-based image assignment
- Admin panel for user & task management
- Progress tracking dashboard

Usage:
    pip install flask psycopg2-binary werkzeug
    python app.py
"""

import os
import sys
import json
import yaml
from datetime import datetime, date
from decimal import Decimal
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError:
    print("ERROR: werkzeug not installed. Run: pip install werkzeug")
    sys.exit(1)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for database types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# App Config
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "olyvision-secret-key-change-in-production")
app.json_encoder = CustomJSONEncoder

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "highlander"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

# MinIO config
MINIO_PUBLIC_URL = os.environ.get("MINIO_PUBLIC_URL", "http://localhost:9000")


# ---------------------------------------------------------------------------
# Database helpers (with Connection Pooling)
# ---------------------------------------------------------------------------

import psycopg2.pool

_db_pool = None

def get_db_pool():
    global _db_pool
    if _db_pool is None:
        try:
            _db_pool = psycopg2.pool.ThreadedConnectionPool(2, 20, **DB_CONFIG)
        except Exception as e:
            print(f"[WARN] Connection pool init error: {e}")
    return _db_pool

class PooledConn:
    """Wrapper that returns connection to pool upon close()."""
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        self._closed = False

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pool and self._conn:
                try:
                    self._pool.putconn(self._conn)
                    return
                except Exception:
                    pass
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._conn.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        return getattr(self._conn, name)

def get_db_connection():
    """Create or retrieve a database connection."""
    pool = get_db_pool()
    if pool:
        try:
            conn = pool.getconn()
            if conn.closed:
                conn = psycopg2.connect(**DB_CONFIG)
            return PooledConn(conn, pool)
        except Exception:
            pass
    return psycopg2.connect(**DB_CONFIG)


def init_database(max_retries=10, retry_delay=2):
    """Create tables, default users, and sample seed data if database is empty."""
    import time
    conn = None
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_db_connection()
            break
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Could not connect to database after {max_retries} attempts: {e}")
                return
            print(f"Waiting for database connection... (attempt {attempt}/{max_retries})")
            time.sleep(retry_delay)

    create_tables_sql = """
    -- Users table
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        email VARCHAR(255),
        password_hash VARCHAR(255) NOT NULL,
        role VARCHAR(50) DEFAULT 'curator',
        is_active BOOLEAN DEFAULT TRUE,
        created_by INTEGER REFERENCES users(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    );

    -- VLM Inference results table
    CREATE TABLE IF NOT EXISTS vlm_inference (
        id SERIAL PRIMARY KEY,
        store_id INTEGER,
        uuid VARCHAR(36) NOT NULL,
        image_url TEXT,
        minio_url TEXT,
        age_category VARCHAR(20),
        gender_category VARCHAR(20),
        is_staff BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_vlm_inference_uuid ON vlm_inference(uuid);
    CREATE INDEX IF NOT EXISTS idx_vlm_inference_store_id ON vlm_inference(store_id);

    -- Curated results table
    CREATE TABLE IF NOT EXISTS vlm_curated (
        id SERIAL PRIMARY KEY,
        inference_id INTEGER REFERENCES vlm_inference(id),
        store_id INTEGER,
        uuid VARCHAR(36) NOT NULL,
        image_url TEXT,
        minio_url TEXT,
        age_category VARCHAR(20),
        gender_category VARCHAR(20),
        is_staff BOOLEAN,
        curated_by INTEGER REFERENCES users(id),
        curated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_vlm_curated_inference_id ON vlm_curated(inference_id);
    CREATE INDEX IF NOT EXISTS idx_vlm_curated_curated_by ON vlm_curated(curated_by);

    -- Batch assignments table
    CREATE TABLE IF NOT EXISTS batch_assignments (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        batch_name VARCHAR(255) NOT NULL,
        image_id_from INTEGER NOT NULL,
        image_id_to INTEGER NOT NULL,
        total_images INTEGER NOT NULL,
        status VARCHAR(50) DEFAULT 'active',
        created_by INTEGER REFERENCES users(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_batch_assignments_user ON batch_assignments(user_id);
    CREATE INDEX IF NOT EXISTS idx_batch_assignments_status ON batch_assignments(status);

    -- Image assignments table
    CREATE TABLE IF NOT EXISTS image_assignments (
        id SERIAL PRIMARY KEY,
        inference_id INTEGER UNIQUE REFERENCES vlm_inference(id),
        batch_id INTEGER REFERENCES batch_assignments(id) ON DELETE CASCADE,
        assigned_to INTEGER REFERENCES users(id),
        status VARCHAR(50) DEFAULT 'pending',
        assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_image_assignments_user ON image_assignments(assigned_to);
    CREATE INDEX IF NOT EXISTS idx_image_assignments_batch ON image_assignments(batch_id);
    CREATE INDEX IF NOT EXISTS idx_image_assignments_status ON image_assignments(status);

    -- Training experiments table
    CREATE TABLE IF NOT EXISTS training_experiments (
        id SERIAL PRIMARY KEY,
        experiment_id VARCHAR(100) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        task VARCHAR(50) NOT NULL,
        description TEXT,
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        data_export_id VARCHAR(100),
        train_images INTEGER,
        valid_images INTEGER,
        test_images INTEGER,
        num_classes INTEGER,
        class_names JSONB,
        model_architecture VARCHAR(100),
        pretrained BOOLEAN DEFAULT TRUE,
        config JSONB,
        best_epoch INTEGER,
        best_val_accuracy DECIMAL(6,4),
        best_val_loss DECIMAL(10,6),
        test_accuracy DECIMAL(6,4),
        test_precision DECIMAL(6,4),
        test_recall DECIMAL(6,4),
        test_f1 DECIMAL(6,4),
        training_time_seconds INTEGER,
        artifact_dir VARCHAR(500),
        best_model_path VARCHAR(500),
        onnx_model_path VARCHAR(500),
        error_message TEXT,
        notes TEXT
    );

    -- Training metrics table
    CREATE TABLE IF NOT EXISTS training_metrics (
        id SERIAL PRIMARY KEY,
        experiment_id VARCHAR(100) REFERENCES training_experiments(experiment_id) ON DELETE CASCADE,
        epoch INTEGER NOT NULL,
        train_loss DECIMAL(10,6),
        train_accuracy DECIMAL(6,4),
        val_loss DECIMAL(10,6),
        val_accuracy DECIMAL(6,4),
        learning_rate DECIMAL(12,10),
        epoch_time_seconds DECIMAL(10,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(experiment_id, epoch)
    );

    -- Model versions table
    CREATE TABLE IF NOT EXISTS model_versions (
        id SERIAL PRIMARY KEY,
        task VARCHAR(50) NOT NULL,
        version VARCHAR(20) NOT NULL,
        experiment_id VARCHAR(100) REFERENCES training_experiments(experiment_id),
        is_active BOOLEAN DEFAULT FALSE,
        deployed_at TIMESTAMP,
        deployed_by INTEGER,
        model_architecture VARCHAR(100),
        num_classes INTEGER,
        class_names JSONB,
        onnx_path VARCHAR(500),
        pt_path VARCHAR(500),
        config_path VARCHAR(500),
        accuracy DECIMAL(6,4),
        precision_score DECIMAL(6,4),
        recall_score DECIMAL(6,4),
        f1_score DECIMAL(6,4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(task, version)
    );
    """
    try:
        with conn.cursor() as cur:
            cur.execute(create_tables_sql)
            
            # Check if admin user exists, create if not
            cur.execute("SELECT id FROM users WHERE username = 'admin'")
            admin_row = cur.fetchone()
            if not admin_row:
                admin_hash = generate_password_hash("admin123")
                cur.execute("""
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES ('admin', 'admin@olyvision.com', %s, 'admin')
                    RETURNING id
                """, (admin_hash,))
                admin_id = cur.fetchone()[0]
                print("[OK] Default admin user created (admin/admin123)")
            else:
                admin_id = admin_row[0]

            # Check if curator user exists, create if not
            cur.execute("SELECT id FROM users WHERE username = 'curator'")
            curator_row = cur.fetchone()
            if not curator_row:
                curator_hash = generate_password_hash("curator123")
                cur.execute("""
                    INSERT INTO users (username, email, password_hash, role, created_by)
                    VALUES ('curator', 'curator@olyvision.com', %s, 'curator', %s)
                    RETURNING id
                """, (curator_hash, admin_id))
                curator_id = cur.fetchone()[0]
                print("[OK] Default curator user created (curator/curator123)")
            else:
                curator_id = curator_row[0]
            
            # Check if actual inference crops should be seeded
            cur.execute("SELECT COUNT(*) FROM vlm_inference")
            count = cur.fetchone()[0]
            if count == 0:
                print("[INFO] Seeding initial pipeline crops from data/exports and data/prepared...")
                webapp_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_dir = os.path.dirname(webapp_dir)
                data_dir = os.path.join(workspace_dir, "data")
                
                records_by_uuid = {}
                import re
                
                for sub in ["exports", "prepared"]:
                    target_folder = os.path.join(data_dir, sub)
                    if not os.path.exists(target_folder):
                        continue
                    for root, dirs, files in os.walk(target_folder):
                        for f in files:
                            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                                continue
                            m = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', f)
                            if not m:
                                continue
                            uuid_str = m.group(1)
                            full_p = os.path.join(root, f)
                            norm_p = full_p.replace(os.sep, '/').lower()
                            rel_p = '/' + os.path.relpath(full_p, workspace_dir).replace(os.sep, '/')
                            
                            if uuid_str not in records_by_uuid:
                                records_by_uuid[uuid_str] = {
                                    'uuid': uuid_str,
                                    'image_url': rel_p,
                                    'gender': 'unknown',
                                    'age': 'adult',
                                    'is_staff': False,
                                    'store_id': 18
                                }
                                
                            if '/male/' in norm_p:
                                records_by_uuid[uuid_str]['gender'] = 'male'
                            elif '/female/' in norm_p:
                                records_by_uuid[uuid_str]['gender'] = 'female'
                                
                            for a in ['infant', 'child', 'teen', 'adult', 'mature']:
                                if f'/{a}/' in norm_p:
                                    records_by_uuid[uuid_str]['age'] = a
                                    break
                                    
                inserted_ids = []
                for r in records_by_uuid.values():
                    cur.execute("""
                        INSERT INTO vlm_inference (store_id, uuid, image_url, age_category, gender_category, is_staff)
                        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                    """, (r['store_id'], r['uuid'], r['image_url'], r['age'], r['gender'], r['is_staff']))
                    inserted_ids.append(cur.fetchone()[0])
                    
                if curator_id and admin_id and inserted_ids:
                    from_id = min(inserted_ids)
                    to_id = max(inserted_ids)
                    cur.execute("""
                        INSERT INTO batch_assignments (user_id, batch_name, image_id_from, image_id_to, total_images, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                    """, (curator_id, "Batch 1 - Live Pipeline Crops", from_id, to_id, len(inserted_ids), admin_id))
                    batch_id = cur.fetchone()[0]
                    for img_id in inserted_ids:
                        cur.execute("""
                            INSERT INTO image_assignments (inference_id, batch_id, assigned_to)
                            VALUES (%s, %s, %s) ON CONFLICT (inference_id) DO NOTHING
                        """, (img_id, batch_id, curator_id))
                print(f"[OK] Seeded {len(inserted_ids)} actual crops and initial batch for curator.")

        conn.commit()
        conn.close()
        print("[OK] Database tables initialized")
    except Exception as e:
        print(f"Database init error: {e}")


def get_user_by_username(username):
    """Get user from database by username."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE username = %s AND is_active = TRUE", (username,))
        user = cur.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    """Get user from database by ID."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
    conn.close()
    return user


# ---------------------------------------------------------------------------
# Auth decorators
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access required", "error")
            return redirect(url_for("main"))
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------------------------------------------------------
# Routes - Authentication
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("main"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        user = get_user_by_username(username)
        
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user"] = user["username"]
            session["role"] = user["role"]
            
            # Update last login
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))
            conn.commit()
            conn.close()
            
            return redirect(url_for("main"))
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes - Main Pages
# ---------------------------------------------------------------------------

@app.route("/main")
@login_required
def main():
    return render_template("main.html", user=session["user"], role=session.get("role", "curator"))


@app.route("/curation")
@login_required
def curation():
    return render_template("curation.html", user=session["user"], role=session.get("role", "curator"))


@app.route("/themes")
@login_required
def themes():
    return render_template("themes.html", user=session["user"], role=session.get("role", "curator"))


@app.route("/theme/<theme_name>")
@login_required
def theme_direct(theme_name):
    return redirect(url_for("curation", theme=theme_name))


@app.route("/admin")
@admin_required
def admin_panel():
    return render_template("admin.html", user=session["user"], role=session.get("role"))


# ---------------------------------------------------------------------------
# API Routes - User Management (Admin only)
# ---------------------------------------------------------------------------

@app.route("/api/admin/users", methods=["GET"])
@admin_required
def get_users():
    """Get all users."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, username, email, role, is_active, created_at, last_login
            FROM users
            ORDER BY id
        """)
        users = cur.fetchall()
    conn.close()
    
    # Convert datetime objects
    for user in users:
        if user.get('created_at'):
            user['created_at'] = user['created_at'].isoformat()
        if user.get('last_login'):
            user['last_login'] = user['last_login'].isoformat()
    
    return jsonify({"users": users})


@app.route("/api/admin/users", methods=["POST"])
@admin_required
def create_user():
    """Create a new user."""
    data = request.json
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    role = data.get("role", "curator")
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    if role not in ["admin", "curator"]:
        return jsonify({"error": "Invalid role"}), 400
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Check if username exists
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return jsonify({"error": "Username already exists"}), 400
            
            password_hash = generate_password_hash(password)
            cur.execute("""
                INSERT INTO users (username, email, password_hash, role, created_by)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (username, email, password_hash, role, session["user_id"]))
            new_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "id": new_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>", methods=["PUT"])
@admin_required
def update_user(user_id):
    """Update a user."""
    data = request.json
    
    updates = []
    params = []
    
    if "email" in data:
        updates.append("email = %s")
        params.append(data["email"])
    
    if "role" in data and data["role"] in ["admin", "curator"]:
        updates.append("role = %s")
        params.append(data["role"])
    
    if "is_active" in data:
        updates.append("is_active = %s")
        params.append(data["is_active"])
    
    if "password" in data and data["password"]:
        updates.append("password_hash = %s")
        params.append(generate_password_hash(data["password"]))
    
    if not updates:
        return jsonify({"error": "No updates provided"}), 400
    
    params.append(user_id)
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", params)
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    """Deactivate a user (soft delete)."""
    if user_id == session["user_id"]:
        return jsonify({"error": "Cannot delete yourself"}), 400
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET is_active = FALSE WHERE id = %s", (user_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API Routes - Batch Assignment (Admin only)
# ---------------------------------------------------------------------------

@app.route("/api/admin/batches", methods=["GET"])
@admin_required
def get_batches():
    """Get all batch assignments."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT ba.*, u.username as assigned_to_name,
                   (SELECT COUNT(*) FROM image_assignments ia 
                    WHERE ia.batch_id = ba.id AND ia.status = 'completed') as actual_completed
            FROM batch_assignments ba
            LEFT JOIN users u ON ba.user_id = u.id
            ORDER BY ba.created_at DESC
        """)
        batches = cur.fetchall()
    conn.close()
    
    for batch in batches:
        if batch.get('created_at'):
            batch['created_at'] = batch['created_at'].isoformat()
        if batch.get('completed_at'):
            batch['completed_at'] = batch['completed_at'].isoformat()
    
    return jsonify({"batches": batches})


@app.route("/api/admin/batches", methods=["POST"])
@admin_required
def create_batch():
    """Create a new batch assignment."""
    data = request.json
    user_id = data.get("user_id")
    batch_name = data.get("batch_name", "")
    image_count = data.get("image_count", 100)
    
    if not user_id:
        return jsonify({"error": "User ID required"}), 400
    
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find next available unassigned images
            cur.execute("""
                SELECT vi.id FROM vlm_inference vi
                LEFT JOIN vlm_curated vc ON vi.id = vc.inference_id
                LEFT JOIN image_assignments ia ON vi.id = ia.inference_id
                WHERE vc.id IS NULL AND ia.id IS NULL
                ORDER BY vi.id
                LIMIT %s
            """, (image_count,))
            available_images = cur.fetchall()
            
            if not available_images:
                return jsonify({"error": "No unassigned images available"}), 400
            
            image_ids = [img["id"] for img in available_images]
            image_id_from = min(image_ids)
            image_id_to = max(image_ids)
            total_images = len(image_ids)
            
            # Create batch assignment
            cur.execute("""
                INSERT INTO batch_assignments 
                (user_id, batch_name, image_id_from, image_id_to, total_images, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (user_id, batch_name or f"Batch {datetime.now().strftime('%Y%m%d_%H%M')}", 
                  image_id_from, image_id_to, total_images, session["user_id"]))
            batch_id = cur.fetchone()["id"]
            
            # Assign individual images (track actual assignments)
            actually_assigned = 0
            for img_id in image_ids:
                cur.execute("""
                    INSERT INTO image_assignments (inference_id, batch_id, assigned_to)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (inference_id) DO NOTHING
                    RETURNING id
                """, (img_id, batch_id, user_id))
                if cur.fetchone():
                    actually_assigned += 1
            
            # Update batch with actual count if different
            if actually_assigned != total_images:
                cur.execute("""
                    UPDATE batch_assignments 
                    SET total_images = %s 
                    WHERE id = %s
                """, (actually_assigned, batch_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True, 
            "batch_id": batch_id, 
            "images_requested": total_images,
            "images_assigned": actually_assigned,
            "note": "Some images may have been skipped if already assigned" if actually_assigned != total_images else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/batches/<int:batch_id>", methods=["PUT"])
@admin_required
def update_batch(batch_id):
    """Update a batch assignment."""
    data = request.json
    status = data.get("status")
    
    if status and status not in ["active", "paused", "completed"]:
        return jsonify({"error": "Invalid status"}), 400
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if status:
                cur.execute("UPDATE batch_assignments SET status = %s WHERE id = %s", (status, batch_id))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/batches/<int:batch_id>", methods=["DELETE"])
@admin_required
def delete_batch(batch_id):
    """Delete a batch assignment and unassign images."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Delete image assignments first
            cur.execute("DELETE FROM image_assignments WHERE batch_id = %s", (batch_id,))
            # Delete batch
            cur.execute("DELETE FROM batch_assignments WHERE id = %s", (batch_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/stats")
@admin_required
def get_admin_stats():
    """Get admin dashboard statistics."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Overall stats
        cur.execute("""
            SELECT 
                (SELECT COUNT(*) FROM vlm_inference) as total_images,
                (SELECT COUNT(*) FROM vlm_curated) as curated_images,
                (SELECT COUNT(*) FROM image_assignments) as assigned_images,
                (SELECT COUNT(*) FROM users WHERE is_active = TRUE) as active_users,
                (SELECT COUNT(*) FROM batch_assignments WHERE status = 'active') as active_batches
        """)
        stats = dict(cur.fetchone())
        
        # Per-user stats
        cur.execute("""
            SELECT u.id, u.username,
                   COUNT(DISTINCT ia.inference_id) as assigned,
                   COUNT(DISTINCT CASE WHEN ia.status = 'completed' THEN ia.inference_id END) as completed
            FROM users u
            LEFT JOIN image_assignments ia ON u.id = ia.assigned_to
            WHERE u.role = 'curator' AND u.is_active = TRUE
            GROUP BY u.id, u.username
            ORDER BY u.username
        """)
        stats["user_stats"] = [dict(row) for row in cur.fetchall()]
    
    conn.close()
    
    stats["unassigned_images"] = stats["total_images"] - stats["assigned_images"]
    stats["pending_images"] = stats["total_images"] - stats["curated_images"]
    
    return jsonify(stats)


@app.route("/api/admin/unassigned-count")
@admin_required
def get_unassigned_count():
    """Get count of unassigned images."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT COUNT(*) as count FROM vlm_inference vi
            LEFT JOIN vlm_curated vc ON vi.id = vc.inference_id
            LEFT JOIN image_assignments ia ON vi.id = ia.inference_id
            WHERE vc.id IS NULL AND ia.id IS NULL
        """)
        result = cur.fetchone()
    conn.close()
    
    return jsonify({"unassigned_count": result["count"]})


@app.route("/api/admin/assignment-health")
@admin_required
def check_assignment_health():
    """Check for any assignment issues (duplicate assignments, etc.)."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check for duplicate assignments (should be 0)
        cur.execute("""
            SELECT inference_id, COUNT(*) as count 
            FROM image_assignments 
            GROUP BY inference_id 
            HAVING COUNT(*) > 1
        """)
        duplicates = cur.fetchall()
        
        # Check for images assigned to multiple batches
        cur.execute("""
            SELECT ia.inference_id, array_agg(ia.batch_id) as batch_ids
            FROM image_assignments ia
            GROUP BY ia.inference_id
            HAVING COUNT(DISTINCT ia.batch_id) > 1
        """)
        multi_batch = cur.fetchall()
        
        # Get overall stats
        cur.execute("""
            SELECT 
                (SELECT COUNT(*) FROM vlm_inference) as total_images,
                (SELECT COUNT(*) FROM image_assignments) as assigned_images,
                (SELECT COUNT(*) FROM vlm_curated) as curated_images,
                (SELECT COUNT(DISTINCT inference_id) FROM image_assignments) as unique_assigned,
                (SELECT COUNT(*) FROM vlm_inference vi
                 LEFT JOIN image_assignments ia ON vi.id = ia.inference_id
                 LEFT JOIN vlm_curated vc ON vi.id = vc.inference_id
                 WHERE ia.id IS NULL AND vc.id IS NULL) as unassigned_uncurated
        """)
        stats = dict(cur.fetchone())
    
    conn.close()
    
    health_status = "healthy" if len(duplicates) == 0 and len(multi_batch) == 0 else "issues_found"
    
    return jsonify({
        "status": health_status,
        "duplicate_assignments": len(duplicates),
        "multi_batch_assignments": len(multi_batch),
        "duplicates_detail": [dict(d) for d in duplicates],
        "multi_batch_detail": [dict(m) for m in multi_batch],
        "stats": stats
    })


# ---------------------------------------------------------------------------
# API Routes - Curator
# ---------------------------------------------------------------------------

@app.route("/api/my-assignments")
@login_required
def get_my_assignments():
    """Get current user's batch assignments."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT ba.*,
                   (SELECT COUNT(*) FROM image_assignments ia 
                    WHERE ia.batch_id = ba.id AND ia.status = 'completed') as actual_completed,
                   (SELECT COUNT(*) FROM image_assignments ia 
                    WHERE ia.batch_id = ba.id AND ia.status = 'pending') as pending
            FROM batch_assignments ba
            WHERE ba.user_id = %s AND ba.status = 'active'
            ORDER BY ba.priority DESC, ba.created_at
        """, (session["user_id"],))
        batches = cur.fetchall()
    conn.close()
    
    for batch in batches:
        if batch.get('created_at'):
            batch['created_at'] = batch['created_at'].isoformat()
        batch['progress'] = round((batch['actual_completed'] / batch['total_images']) * 100, 1) if batch['total_images'] > 0 else 0
    
    return jsonify({"assignments": batches})


@app.route("/api/my-stats")
@login_required
def get_my_stats():
    """Get current user's statistics."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE ia.status = 'pending') as pending,
                COUNT(*) FILTER (WHERE ia.status = 'completed') as completed,
                COUNT(*) as total_assigned
            FROM image_assignments ia
            WHERE ia.assigned_to = %s
        """, (session["user_id"],))
        stats = dict(cur.fetchone())
        
        # Today's progress
        cur.execute("""
            SELECT COUNT(*) as today_completed
            FROM image_assignments ia
            WHERE ia.assigned_to = %s 
            AND ia.status = 'completed'
            AND ia.completed_at::date = CURRENT_DATE
        """, (session["user_id"],))
        stats["today_completed"] = cur.fetchone()["today_completed"]
    
    conn.close()
    
    return jsonify(stats)


@app.route("/api/images")
@login_required
def get_images():
    """Get images for curation (filtered by assignment for curators)."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    offset = (page - 1) * per_page
    
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # For curators, only show their assigned images
        # For admins, show all uncurated images
        if session.get("role") == "admin":
            cur.execute("""
                SELECT vi.id, vi.store_id, vi.uuid, vi.image_url, vi.minio_url,
                       vi.age_category, vi.gender_category, vi.is_staff, vi.created_at
                FROM vlm_inference vi
                LEFT JOIN vlm_curated vc ON vi.id = vc.inference_id
                WHERE vc.id IS NULL
                ORDER BY vi.id
                LIMIT %s OFFSET %s
            """, (per_page, offset))
        else:
            # Curator - only assigned images
            cur.execute("""
                SELECT vi.id, vi.store_id, vi.uuid, vi.image_url, vi.minio_url,
                       vi.age_category, vi.gender_category, vi.is_staff, vi.created_at,
                       ia.id as assignment_id
                FROM vlm_inference vi
                INNER JOIN image_assignments ia ON vi.id = ia.inference_id
                WHERE ia.assigned_to = %s AND ia.status = 'pending'
                ORDER BY vi.id
                LIMIT %s OFFSET %s
            """, (session["user_id"], per_page, offset))
        
        rows = cur.fetchall()
        
        images = []
        for row in rows:
            img = dict(row)
            if img.get('created_at'):
                img['created_at'] = img['created_at'].isoformat()
            images.append(img)
        
        # Get total count
        if session.get("role") == "admin":
            cur.execute("""
                SELECT COUNT(*) as total
                FROM vlm_inference vi
                LEFT JOIN vlm_curated vc ON vi.id = vc.inference_id
                WHERE vc.id IS NULL
            """)
        else:
            cur.execute("""
                SELECT COUNT(*) as total
                FROM image_assignments ia
                WHERE ia.assigned_to = %s AND ia.status = 'pending'
            """, (session["user_id"],))
        
        total = cur.fetchone()["total"]
    
    conn.close()
    
    return jsonify({
        "images": images,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if total > 0 else 0
    })


@app.route("/api/image/<path:image_path>")
@login_required
def serve_image(image_path):
    """Serve an image file."""
    clean_path = image_path.lstrip("/\\")
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(webapp_dir)
    
    possible_paths = [
        os.path.normpath(os.path.join(workspace_root, clean_path)),
        os.path.normpath(os.path.join(workspace_root, "data", clean_path)),
        os.path.normpath(os.path.join(webapp_dir, clean_path)),
        os.path.normpath(os.path.join(webapp_dir, "data", clean_path)),
        os.path.normpath(clean_path),
    ]
    if clean_path.startswith("data/") or clean_path.startswith("data\\"):
        sub_path = clean_path[5:]
        possible_paths.append(os.path.normpath(os.path.join(workspace_root, "data", sub_path)))

    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            res = send_file(os.path.abspath(p), mimetype="image/jpeg", max_age=86400)
            res.headers["Cache-Control"] = "public, max-age=86400, immutable"
            return res
    return "Image not found", 404


@app.route("/api/curate", methods=["POST"])
@login_required
def save_curation():
    """Save curated data."""
    data = request.json
    
    inference_id = data.get("inference_id")
    store_id = data.get("store_id")
    uuid = data.get("uuid")
    image_url = data.get("image_url")
    minio_url = data.get("minio_url")
    age_category = data.get("age_category")
    gender_category = data.get("gender_category")
    is_staff = data.get("is_staff", False)
    
    if not inference_id or not uuid:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Save curation
            cur.execute("""
                INSERT INTO vlm_curated 
                (inference_id, store_id, uuid, image_url, minio_url, age_category, gender_category, is_staff, curated_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (inference_id, store_id, uuid, image_url, minio_url, age_category, gender_category, is_staff, session["user"]))
            new_id = cur.fetchone()[0]
            
            # Update image assignment status
            cur.execute("""
                UPDATE image_assignments 
                SET status = 'completed', completed_at = NOW()
                WHERE inference_id = %s AND assigned_to = %s
            """, (inference_id, session["user_id"]))
            
            # Update batch completed count
            cur.execute("""
                UPDATE batch_assignments ba
                SET completed_count = (
                    SELECT COUNT(*) FROM image_assignments ia 
                    WHERE ia.batch_id = ba.id AND ia.status = 'completed'
                )
                WHERE ba.id IN (
                    SELECT batch_id FROM image_assignments WHERE inference_id = %s
                )
            """, (inference_id,))
            
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "id": new_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
@login_required
def get_stats():
    """Get curation statistics."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                (SELECT COUNT(*) FROM vlm_inference) as total_images,
                (SELECT COUNT(*) FROM vlm_curated) as curated_images,
                (SELECT COUNT(DISTINCT curated_by) FROM vlm_curated) as curators
        """)
        stats = dict(cur.fetchone())
    conn.close()
    
    stats["pending_images"] = stats["total_images"] - stats["curated_images"]
    return jsonify(stats)


@app.route("/api/leaderboard")
@login_required
def get_leaderboard():
    """Get today's curation leaderboard."""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get today's top curators
        cur.execute("""
            SELECT 
                vc.curated_by as username,
                COUNT(*) as count
            FROM vlm_curated vc
            WHERE vc.curated_at::date = CURRENT_DATE
            GROUP BY vc.curated_by
            ORDER BY count DESC
            LIMIT 10
        """)
        today = [dict(row) for row in cur.fetchall()]
        
        # Get all-time top curators
        cur.execute("""
            SELECT 
                vc.curated_by as username,
                COUNT(*) as count
            FROM vlm_curated vc
            GROUP BY vc.curated_by
            ORDER BY count DESC
            LIMIT 10
        """)
        all_time = [dict(row) for row in cur.fetchall()]
        
        # Get current user's stats
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE curated_at::date = CURRENT_DATE) as today,
                COUNT(*) as total
            FROM vlm_curated
            WHERE curated_by = %s
        """, (session["user"],))
        my_stats = dict(cur.fetchone())
    
    conn.close()
    
    # Add rank medals
    medals = ['🥇', '🥈', '🥉']
    for i, item in enumerate(today):
        item['rank'] = medals[i] if i < 3 else str(i + 1)
        item['is_me'] = item['username'] == session["user"]
    
    for i, item in enumerate(all_time):
        item['rank'] = medals[i] if i < 3 else str(i + 1)
        item['is_me'] = item['username'] == session["user"]
    
    return jsonify({
        "today": today,
        "all_time": all_time,
        "my_stats": my_stats,
        "current_user": session["user"]
    })


@app.route("/api/curate/<int:curation_id>", methods=["DELETE"])
@login_required
def undo_curation(curation_id):
    """Undo a curation (delete it)."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify ownership
            cur.execute("""
                SELECT id, inference_id FROM vlm_curated 
                WHERE id = %s AND curated_by = %s
            """, (curation_id, session["user"]))
            curation = cur.fetchone()
            
            if not curation:
                return jsonify({"error": "Curation not found or not yours"}), 404
            
            # Delete curation
            cur.execute("DELETE FROM vlm_curated WHERE id = %s", (curation_id,))
            
            # Reset image assignment status
            cur.execute("""
                UPDATE image_assignments 
                SET status = 'pending', completed_at = NULL
                WHERE inference_id = %s
            """, (curation["inference_id"],))
            
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Training Management Routes
# ---------------------------------------------------------------------------

# Add model_training to path
import subprocess
import threading
MODEL_TRAINING_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_training")
sys.path.insert(0, MODEL_TRAINING_PATH)

# Training job status storage (in-memory, could be moved to DB)
training_jobs = {}


@app.route("/training")
@admin_required
def training_dashboard():
    """Training management dashboard."""
    return render_template("training.html", user=session.get("user"), role=session.get("role"))


@app.route("/api/training/stats")
@admin_required
def get_training_stats():
    """Get training statistics and curated data counts."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Count curated data by task (excluding unknown)
            stats = {}
            
            # Gender stats
            cur.execute("""
                SELECT gender_category as label, COUNT(*) as count 
                FROM vlm_curated 
                WHERE gender_category IN ('male', 'female')
                GROUP BY gender_category
            """)
            gender_counts = {row['label']: row['count'] for row in cur.fetchall()}
            stats['gender'] = {
                'total': sum(gender_counts.values()),
                'classes': gender_counts,
                'ready': sum(gender_counts.values()) >= 20
            }
            
            # Age stats
            cur.execute("""
                SELECT age_category as label, COUNT(*) as count 
                FROM vlm_curated 
                WHERE age_category IN ('infant', 'child', 'teen', 'adult', 'mature')
                GROUP BY age_category
            """)
            age_counts = {row['label']: row['count'] for row in cur.fetchall()}
            stats['age'] = {
                'total': sum(age_counts.values()),
                'classes': age_counts,
                'ready': sum(age_counts.values()) >= 20
            }
            
            # Staff stats
            cur.execute("""
                SELECT 
                    CASE WHEN is_staff THEN 'staff' ELSE 'non_staff' END as label,
                    COUNT(*) as count 
                FROM vlm_curated 
                WHERE is_staff IS NOT NULL
                GROUP BY is_staff
            """)
            staff_counts = {row['label']: row['count'] for row in cur.fetchall()}
            stats['staff'] = {
                'total': sum(staff_counts.values()),
                'classes': staff_counts,
                'ready': sum(staff_counts.values()) >= 20
            }
            
        conn.close()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/experiments")
@admin_required
def get_experiments():
    """Get list of training experiments."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'training_experiments'
                )
            """)
            if not cur.fetchone()['exists']:
                return jsonify([])
            
            task = request.args.get('task')
            status = request.args.get('status')
            
            sql = "SELECT * FROM training_experiments WHERE 1=1"
            params = []
            
            if task:
                sql += " AND task = %s"
                params.append(task)
            if status:
                sql += " AND status = %s"
                params.append(status)
            
            sql += " ORDER BY created_at DESC LIMIT 50"
            
            cur.execute(sql, params)
            experiments = cur.fetchall()
            
        conn.close()
        return jsonify(experiments)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/experiments/<experiment_id>")
@admin_required
def get_experiment_details(experiment_id):
    """Get details of a specific experiment."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get experiment
            cur.execute("SELECT * FROM training_experiments WHERE experiment_id = %s", (experiment_id,))
            experiment = cur.fetchone()
            
            if not experiment:
                return jsonify({"error": "Experiment not found"}), 404
            
            # Get metrics
            cur.execute("""
                SELECT * FROM training_metrics 
                WHERE experiment_id = %s 
                ORDER BY epoch
            """, (experiment_id,))
            metrics = cur.fetchall()
            
            experiment['metrics'] = metrics
            
        conn.close()
        return jsonify(experiment)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/models")
@admin_required
def get_model_versions():
    """Get list of deployed model versions."""
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_versions'
                )
            """)
            if not cur.fetchone()['exists']:
                return jsonify([])
            
            task = request.args.get('task')
            
            sql = "SELECT * FROM model_versions"
            params = []
            
            if task:
                sql += " WHERE task = %s"
                params.append(task)
            
            sql += " ORDER BY created_at DESC"
            
            cur.execute(sql, params)
            models = cur.fetchall()
            
        conn.close()
        return jsonify(models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/start", methods=["POST"])
@admin_required
def start_training():
    """Start a new training job."""
    try:
        data = request.json
        task = data.get('task')
        
        if task not in ['gender', 'age', 'staff']:
            return jsonify({"error": "Invalid task"}), 400
        
        # Generate job ID
        job_id = f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if already running
        for jid, job in training_jobs.items():
            if job.get('status') == 'running' and job.get('task') == task:
                return jsonify({"error": f"Training already running for {task}"}), 400
        
        # Build command
        cmd = [
            sys.executable,
            os.path.join(MODEL_TRAINING_PATH, "run_pipeline.py"),
            "--task", task,
            "--output-base", os.path.dirname(MODEL_TRAINING_PATH),
        ]
        
        if data.get('no_augment'):
            cmd.append("--no-augment")
        
        if data.get('deploy'):
            cmd.append("--deploy")
        
        # Add model architecture if specified
        model = data.get('model', '').strip()
        if model:
            cmd.extend(["--model", model])
        
        # Start training in background
        def run_training():
            training_jobs[job_id]['status'] = 'running'
            training_jobs[job_id]['started_at'] = datetime.now().isoformat()
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=MODEL_TRAINING_PATH
                )
                
                training_jobs[job_id]['pid'] = process.pid
                output_lines = []
                
                for line in process.stdout:
                    output_lines.append(line)
                    training_jobs[job_id]['output'] = ''.join(output_lines[-100:])  # Keep last 100 lines
                
                process.wait()
                
                if process.returncode == 0:
                    training_jobs[job_id]['status'] = 'completed'
                else:
                    training_jobs[job_id]['status'] = 'failed'
                    training_jobs[job_id]['error'] = f"Exit code: {process.returncode}"
                    
            except Exception as e:
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = str(e)
            
            training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        # Initialize job
        training_jobs[job_id] = {
            'id': job_id,
            'task': task,
            'status': 'starting',
            'created_at': datetime.now().isoformat(),
            'output': '',
            'config': data,
        }
        
        # Start thread
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"Training started for {task}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/jobs")
@admin_required
def get_training_jobs():
    """Get status of training jobs."""
    return jsonify(list(training_jobs.values()))


@app.route("/api/training/jobs/<job_id>")
@admin_required
def get_training_job(job_id):
    """Get status of a specific training job."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(training_jobs[job_id])


@app.route("/api/training/deploy", methods=["POST"])
@admin_required
def deploy_model():
    """Deploy a model version as active."""
    try:
        data = request.json
        task = data.get('task')
        version = data.get('version')
        
        if not task or not version:
            return jsonify({"error": "Task and version required"}), 400
        
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Deactivate all versions for task
            cur.execute("UPDATE model_versions SET is_active = FALSE WHERE task = %s", (task,))
            
            # Activate specified version
            cur.execute("""
                UPDATE model_versions 
                SET is_active = TRUE, deployed_at = CURRENT_TIMESTAMP, deployed_by = %s
                WHERE task = %s AND version = %s
            """, (session.get('user_id'), task, version))
            
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": f"Model {version} deployed for {task}"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Artifact Viewing Routes
# ---------------------------------------------------------------------------

ARTIFACTS_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


@app.route("/api/training/experiments/<experiment_id>/artifacts")
@admin_required
def get_experiment_artifacts(experiment_id):
    """Get all artifacts for an experiment including config, metrics, and file paths."""
    try:
        # Find experiment directory
        artifact_dir = os.path.join(ARTIFACTS_BASE_PATH, experiment_id)
        
        if not os.path.isdir(artifact_dir):
            return jsonify({"error": "Experiment artifacts not found"}), 404
        
        result = {
            "experiment_id": experiment_id,
            "artifact_dir": artifact_dir,
            "config": None,
            "class_names": None,
            "classification_report": None,
            "files": [],
            "plots": []
        }
        
        # Load config.yaml
        config_path = os.path.join(artifact_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                result["config"] = yaml.safe_load(f)
        
        # Load class_names.json
        class_names_path = os.path.join(artifact_dir, "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                result["class_names"] = json.load(f)
        
        # Load classification_report.json
        report_path = os.path.join(artifact_dir, "classification_report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                result["classification_report"] = json.load(f)
        
        # List all files
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, artifact_dir)
                file_info = {
                    "name": file,
                    "path": rel_path,
                    "size": os.path.getsize(file_path),
                    "type": file.split('.')[-1] if '.' in file else 'unknown'
                }
                
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    result["plots"].append(file_info)
                else:
                    result["files"].append(file_info)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/artifacts/<path:artifact_path>")
@admin_required
def serve_artifact(artifact_path):
    """Serve an artifact file (image, model, config)."""
    try:
        full_path = os.path.join(ARTIFACTS_BASE_PATH, artifact_path)
        
        # Security check - ensure path is within artifacts directory
        if not os.path.abspath(full_path).startswith(os.path.abspath(ARTIFACTS_BASE_PATH)):
            return jsonify({"error": "Invalid path"}), 403
        
        if not os.path.exists(full_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(full_path)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/artifacts-list")
@admin_required
def list_all_artifacts():
    """List all experiment artifact directories."""
    try:
        experiments = []
        
        if not os.path.isdir(ARTIFACTS_BASE_PATH):
            return jsonify([])
        
        for exp_dir in os.listdir(ARTIFACTS_BASE_PATH):
            exp_path = os.path.join(ARTIFACTS_BASE_PATH, exp_dir)
            if os.path.isdir(exp_path):
                # Try to extract info from directory name
                parts = exp_dir.split('_')
                task = parts[0] if parts else 'unknown'
                
                # Check for key files
                has_onnx = any(f.endswith('.onnx') for f in os.listdir(exp_path))
                has_config = os.path.exists(os.path.join(exp_path, 'config.yaml'))
                
                # Get config if exists
                config = None
                config_path = os.path.join(exp_path, 'config.yaml')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    except:
                        pass
                
                # Get classification report if exists
                metrics = None
                report_path = os.path.join(exp_path, 'classification_report.json')
                if os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            metrics = json.load(f)
                    except:
                        pass
                
                experiments.append({
                    "experiment_id": exp_dir,
                    "task": task,
                    "model": config.get('model', {}).get('architecture') if config else None,
                    "accuracy": metrics.get('accuracy') if metrics else None,
                    "has_onnx": has_onnx,
                    "has_config": has_config,
                    "created": datetime.fromtimestamp(os.path.getctime(exp_path)).isoformat(),
                    "config_summary": {
                        "epochs": config.get('training', {}).get('epochs') if config else None,
                        "learning_rate": config.get('training', {}).get('learning_rate') if config else None,
                        "batch_size": config.get('data', {}).get('batch_size') if config else None,
                        "optimizer": config.get('training', {}).get('optimizer') if config else None,
                    } if config else None
                })
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify(experiments)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_database()
    print("=" * 60)
    print("OLY VISION - Curation Web App (High-Performance Engine)")
    print("=" * 60)
    print("  URL: http://127.0.0.1:5000 / http://localhost:5000")
    print("  Default Admin: admin / admin123")
    print("  Default Curator: curator / curator123")
    print("=" * 60)
    try:
        from waitress import serve
        print("  Engine: Multi-threaded Waitress WSGI (16 threads)")
        serve(app, listen="*:5000", threads=16, channel_timeout=60)
    except ImportError:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
