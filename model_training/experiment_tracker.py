#!/usr/bin/env python3
"""
OLY VISION - Experiment Tracker
--------------------------------
Track training experiments in PostgreSQL database.

Features:
- Create/update experiments
- Log training metrics per epoch
- Store model versions
- Query experiment history
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "highlander",
    "user": "postgres",
    "password": "postgres",
}


def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)


def init_tracking_tables():
    """Create experiment tracking tables if they don't exist."""
    create_tables_sql = """
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
        
        -- Data info
        data_export_id VARCHAR(100),
        train_images INTEGER,
        valid_images INTEGER,
        test_images INTEGER,
        num_classes INTEGER,
        class_names JSONB,
        
        -- Model config
        model_architecture VARCHAR(100),
        pretrained BOOLEAN DEFAULT TRUE,
        
        -- Training config (full config as JSONB)
        config JSONB,
        
        -- Results
        best_epoch INTEGER,
        best_val_accuracy DECIMAL(6,4),
        best_val_loss DECIMAL(10,6),
        test_accuracy DECIMAL(6,4),
        test_precision DECIMAL(6,4),
        test_recall DECIMAL(6,4),
        test_f1 DECIMAL(6,4),
        training_time_seconds INTEGER,
        
        -- Artifact paths
        artifact_dir VARCHAR(500),
        best_model_path VARCHAR(500),
        onnx_model_path VARCHAR(500),
        
        -- Error info
        error_message TEXT,
        
        -- Notes
        notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_experiments_task ON training_experiments(task);
    CREATE INDEX IF NOT EXISTS idx_experiments_status ON training_experiments(status);
    CREATE INDEX IF NOT EXISTS idx_experiments_created ON training_experiments(created_at);

    -- Training metrics per epoch
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

    CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON training_metrics(experiment_id);

    -- Model versions for production deployment
    CREATE TABLE IF NOT EXISTS model_versions (
        id SERIAL PRIMARY KEY,
        task VARCHAR(50) NOT NULL,
        version VARCHAR(20) NOT NULL,
        experiment_id VARCHAR(100) REFERENCES training_experiments(experiment_id),
        is_active BOOLEAN DEFAULT FALSE,
        deployed_at TIMESTAMP,
        deployed_by INTEGER,
        
        -- Model info
        model_architecture VARCHAR(100),
        num_classes INTEGER,
        class_names JSONB,
        
        -- Paths
        onnx_path VARCHAR(500),
        pt_path VARCHAR(500),
        config_path VARCHAR(500),
        
        -- Performance
        accuracy DECIMAL(6,4),
        precision_score DECIMAL(6,4),
        recall_score DECIMAL(6,4),
        
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(task, version)
    );

    CREATE INDEX IF NOT EXISTS idx_versions_task ON model_versions(task);
    CREATE INDEX IF NOT EXISTS idx_versions_active ON model_versions(is_active);
    """
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(create_tables_sql)
        conn.commit()
        print("✓ Experiment tracking tables initialized")
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        conn.close()


class ExperimentTracker:
    """Track training experiments."""
    
    def __init__(self):
        self.conn = None
    
    def _get_conn(self):
        """Get or create database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = get_db_connection()
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        task: str,
        config: Dict[str, Any],
        description: str = "",
        class_names: List[str] = None,
        train_images: int = 0,
        valid_images: int = 0,
        test_images: int = 0,
    ) -> str:
        """Create a new experiment record."""
        conn = self._get_conn()
        
        sql = """
        INSERT INTO training_experiments (
            experiment_id, name, task, description, status, config,
            model_architecture, pretrained, class_names, num_classes,
            train_images, valid_images, test_images
        ) VALUES (
            %s, %s, %s, %s, 'pending', %s,
            %s, %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (experiment_id) DO UPDATE SET
            name = EXCLUDED.name,
            config = EXCLUDED.config,
            status = 'pending'
        RETURNING experiment_id
        """
        
        model_config = config.get('model', {})
        
        with conn.cursor() as cur:
            cur.execute(sql, (
                experiment_id,
                name,
                task,
                description,
                Json(config),
                model_config.get('architecture', 'unknown'),
                model_config.get('pretrained', True),
                Json(class_names) if class_names else None,
                len(class_names) if class_names else 0,
                train_images,
                valid_images,
                test_images,
            ))
            result = cur.fetchone()
        
        conn.commit()
        return result[0]
    
    def start_experiment(self, experiment_id: str):
        """Mark experiment as started."""
        conn = self._get_conn()
        
        sql = """
        UPDATE training_experiments
        SET status = 'running', started_at = CURRENT_TIMESTAMP
        WHERE experiment_id = %s
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, (experiment_id,))
        conn.commit()
    
    def log_epoch(
        self,
        experiment_id: str,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float = None,
        epoch_time: float = None,
    ):
        """Log metrics for an epoch."""
        conn = self._get_conn()
        
        sql = """
        INSERT INTO training_metrics (
            experiment_id, epoch, train_loss, train_accuracy,
            val_loss, val_accuracy, learning_rate, epoch_time_seconds
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (experiment_id, epoch) DO UPDATE SET
            train_loss = EXCLUDED.train_loss,
            train_accuracy = EXCLUDED.train_accuracy,
            val_loss = EXCLUDED.val_loss,
            val_accuracy = EXCLUDED.val_accuracy,
            learning_rate = EXCLUDED.learning_rate,
            epoch_time_seconds = EXCLUDED.epoch_time_seconds
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, (
                experiment_id, epoch, train_loss, train_accuracy,
                val_loss, val_accuracy, learning_rate, epoch_time
            ))
        conn.commit()
    
    def complete_experiment(
        self,
        experiment_id: str,
        best_epoch: int,
        best_val_accuracy: float,
        best_val_loss: float,
        test_accuracy: float = None,
        test_precision: float = None,
        test_recall: float = None,
        test_f1: float = None,
        training_time_seconds: int = None,
        artifact_dir: str = None,
        best_model_path: str = None,
        onnx_model_path: str = None,
    ):
        """Mark experiment as completed with results."""
        conn = self._get_conn()
        
        sql = """
        UPDATE training_experiments SET
            status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            best_epoch = %s,
            best_val_accuracy = %s,
            best_val_loss = %s,
            test_accuracy = %s,
            test_precision = %s,
            test_recall = %s,
            test_f1 = %s,
            training_time_seconds = %s,
            artifact_dir = %s,
            best_model_path = %s,
            onnx_model_path = %s
        WHERE experiment_id = %s
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, (
                best_epoch, best_val_accuracy, best_val_loss,
                test_accuracy, test_precision, test_recall, test_f1,
                training_time_seconds, artifact_dir, best_model_path, onnx_model_path,
                experiment_id
            ))
        conn.commit()
    
    def fail_experiment(self, experiment_id: str, error_message: str):
        """Mark experiment as failed."""
        conn = self._get_conn()
        
        sql = """
        UPDATE training_experiments SET
            status = 'failed',
            completed_at = CURRENT_TIMESTAMP,
            error_message = %s
        WHERE experiment_id = %s
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, (error_message, experiment_id))
        conn.commit()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment details."""
        conn = self._get_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM training_experiments WHERE experiment_id = %s",
                (experiment_id,)
            )
            return cur.fetchone()
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict]:
        """Get all metrics for an experiment."""
        conn = self._get_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM training_metrics WHERE experiment_id = %s ORDER BY epoch",
                (experiment_id,)
            )
            return cur.fetchall()
    
    def list_experiments(
        self,
        task: str = None,
        status: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """List experiments with optional filters."""
        conn = self._get_conn()
        
        sql = "SELECT * FROM training_experiments WHERE 1=1"
        params = []
        
        if task:
            sql += " AND task = %s"
            params.append(task)
        if status:
            sql += " AND status = %s"
            params.append(status)
        
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    
    def create_model_version(
        self,
        task: str,
        version: str,
        experiment_id: str,
        onnx_path: str,
        pt_path: str = None,
        config_path: str = None,
        class_names: List[str] = None,
        accuracy: float = None,
        precision_score: float = None,
        recall_score: float = None,
        notes: str = None,
    ) -> int:
        """Create a new model version."""
        conn = self._get_conn()
        
        # Get experiment info
        experiment = self.get_experiment(experiment_id)
        
        sql = """
        INSERT INTO model_versions (
            task, version, experiment_id, onnx_path, pt_path, config_path,
            model_architecture, num_classes, class_names,
            accuracy, precision_score, recall_score, notes
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s
        )
        RETURNING id
        """
        
        with conn.cursor() as cur:
            cur.execute(sql, (
                task, version, experiment_id, onnx_path, pt_path, config_path,
                experiment['model_architecture'] if experiment else None,
                len(class_names) if class_names else None,
                Json(class_names) if class_names else None,
                accuracy, precision_score, recall_score, notes
            ))
            result = cur.fetchone()
        
        conn.commit()
        return result[0]
    
    def activate_model_version(self, task: str, version: str):
        """Set a model version as active (deactivate others)."""
        conn = self._get_conn()
        
        with conn.cursor() as cur:
            # Deactivate all versions for this task
            cur.execute(
                "UPDATE model_versions SET is_active = FALSE WHERE task = %s",
                (task,)
            )
            # Activate specified version
            cur.execute(
                """UPDATE model_versions 
                   SET is_active = TRUE, deployed_at = CURRENT_TIMESTAMP 
                   WHERE task = %s AND version = %s""",
                (task, version)
            )
        conn.commit()
    
    def get_active_model(self, task: str) -> Optional[Dict]:
        """Get the currently active model for a task."""
        conn = self._get_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM model_versions WHERE task = %s AND is_active = TRUE",
                (task,)
            )
            return cur.fetchone()
    
    def list_model_versions(self, task: str = None) -> List[Dict]:
        """List model versions."""
        conn = self._get_conn()
        
        sql = "SELECT * FROM model_versions"
        params = []
        
        if task:
            sql += " WHERE task = %s"
            params.append(task)
        
        sql += " ORDER BY created_at DESC"
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def generate_experiment_id(task: str, model: str) -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task}_{model}_{timestamp}"


if __name__ == "__main__":
    # Initialize tables
    print("Initializing experiment tracking tables...")
    init_tracking_tables()
    
    # Test tracker
    tracker = ExperimentTracker()
    
    # List recent experiments
    experiments = tracker.list_experiments(limit=5)
    print(f"\nRecent experiments: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp['experiment_id']}: {exp['status']}")
    
    tracker.close()
