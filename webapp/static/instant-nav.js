/**
 * OLY VISION — Elite Instant Smooth Navigation & Zero-Glitch Sidebar Controller
 * Eliminates screen flashes and sidebar layout shifts during page transitions.
 */

(function() {
    // 1. Immediately lock HTML to Elite Dark Gradient Canvas and disable transitions during initial render
    if (document.documentElement) {
        document.documentElement.style.backgroundColor = '#080c14';
        document.documentElement.style.colorScheme = 'dark';
        document.documentElement.classList.add('preload');
    }
    if (document.body) {
        document.body.style.backgroundColor = '#080c14';
    }

    // 2. Prefetch internal links on hover/touch for instantaneous response
    const prefetched = new Set();
    function prefetchUrl(url) {
        if (!url || prefetched.has(url) || url.startsWith('#') || url.startsWith('javascript:')) return;
        try {
            const parsed = new URL(url, window.location.origin);
            if (parsed.origin !== window.location.origin) return;
            if (parsed.pathname === window.location.pathname) return;

            prefetched.add(url);
            const link = document.createElement('link');
            link.rel = 'prefetch';
            link.href = parsed.href;
            document.head.appendChild(link);
        } catch (e) {}
    }

    document.addEventListener('mouseover', (e) => {
        const a = e.target.closest('a');
        if (a && a.href) prefetchUrl(a.href);
    }, { passive: true });

    document.addEventListener('touchstart', (e) => {
        const a = e.target.closest('a');
        if (a && a.href) prefetchUrl(a.href);
    }, { passive: true });

    // 3. Immediately evaluate and apply sidebar shrink state and width synchronously BEFORE paint
    const DEFAULT_SIDEBAR_WIDTH = '138px';
    const MAX_SIDEBAR_WIDTH = 145;
    const MIN_SIDEBAR_WIDTH = 120;
    const SHRINK_BREAKPOINT = 900;

    function getSanitizedWidth() {
        try {
            const raw = localStorage.getItem('oly_sidebar_width');
            if (!raw) return DEFAULT_SIDEBAR_WIDTH;
            const num = parseInt(raw, 10);
            if (isNaN(num) || num > MAX_SIDEBAR_WIDTH || num < MIN_SIDEBAR_WIDTH) {
                localStorage.setItem('oly_sidebar_width', DEFAULT_SIDEBAR_WIDTH);
                return DEFAULT_SIDEBAR_WIDTH;
            }
            return raw;
        } catch (e) {
            return DEFAULT_SIDEBAR_WIDTH;
        }
    }

    try {
        const saved = localStorage.getItem('oly_sidebar_shrunk');
        const savedWidth = getSanitizedWidth();
        const isNarrow = window.innerWidth <= SHRINK_BREAKPOINT;

        if (saved === 'true' || (saved === null && isNarrow) || isNarrow) {
            document.documentElement.classList.add('sidebar-shrunk');
        } else {
            document.documentElement.classList.remove('sidebar-shrunk');
            document.documentElement.style.setProperty('--sidebar-width', savedWidth);
        }
    } catch (e) {}

    // 4. Controller for smooth sidebar interactions & hold-and-move resizing
    function initSidebarController() {
        // Sync body class with html class
        if (document.documentElement.classList.contains('sidebar-shrunk')) {
            document.body.classList.add('sidebar-shrunk');
        } else {
            document.body.classList.remove('sidebar-shrunk');
            const savedWidth = getSanitizedWidth();
            document.documentElement.style.setProperty('--sidebar-width', savedWidth);
        }

        // Lift preload class smoothly on next animation frames so transitions only happen on user action
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                document.documentElement.classList.remove('preload');
            });
        });

        // Setup Hold-and-Move Draggable Resizer Handle on Sidebar
        const sidebar = document.querySelector('.suite-sidebar');
        if (sidebar) {
            // Ensure shrunk logo badge exists in brand
            const brand = sidebar.querySelector('.suite-sidebar-brand');
            if (brand && !brand.querySelector('.suite-brand-shrunk')) {
                const shrunk = document.createElement('div');
                shrunk.className = 'suite-brand-shrunk';
                shrunk.setAttribute('aria-hidden', 'true');
                shrunk.innerHTML = '<span class="shrunk-ml">ML</span><span class="shrunk-ops">ops</span>';
                brand.appendChild(shrunk);
                if (!brand.hasAttribute('data-tooltip')) {
                    brand.setAttribute('data-tooltip', 'MLops curation');
                }
            }

            let resizer = sidebar.querySelector('.suite-sidebar-resizer');
            if (!resizer) {
                resizer = document.createElement('div');
                resizer.className = 'suite-sidebar-resizer';
                resizer.id = 'suiteSidebarResizer';
                resizer.setAttribute('title', 'Hold and drag to shrink or expand sidebar');
                sidebar.appendChild(resizer);
            }

            let isResizing = false;

            function onPointerDown(e) {
                if (e.button !== undefined && e.button !== 0) return; // Left-click only
                isResizing = true;
                document.body.classList.add('is-resizing');
                document.documentElement.classList.add('is-resizing');
                try {
                    localStorage.setItem('oly_sidebar_manual', 'true');
                } catch (err) {}
                e.preventDefault();
                e.stopPropagation();
            }

            function onPointerMove(e) {
                if (!isResizing) return;
                const clientX = e.touches ? e.touches[0].clientX : e.clientX;

                // If dragged left past threshold (95px), snap into shrunk mode
                if (clientX < 95) {
                    if (!document.body.classList.contains('sidebar-shrunk')) {
                        document.body.classList.add('sidebar-shrunk');
                        document.documentElement.classList.add('sidebar-shrunk');
                        try {
                            localStorage.setItem('oly_sidebar_shrunk', 'true');
                        } catch (err) {}
                    }
                } else {
                    // Dragged right past threshold: expand fluidly up to compact max
                    if (document.body.classList.contains('sidebar-shrunk')) {
                        document.body.classList.remove('sidebar-shrunk');
                        document.documentElement.classList.remove('sidebar-shrunk');
                        try {
                            localStorage.setItem('oly_sidebar_shrunk', 'false');
                        } catch (err) {}
                    }
                    const clamped = Math.max(MIN_SIDEBAR_WIDTH, Math.min(clientX, MAX_SIDEBAR_WIDTH));
                    document.documentElement.style.setProperty('--sidebar-width', clamped + 'px');
                    try {
                        localStorage.setItem('oly_sidebar_width', clamped + 'px');
                    } catch (err) {}
                }
            }

            function onPointerUp() {
                if (!isResizing) return;
                isResizing = false;
                document.body.classList.remove('is-resizing');
                document.documentElement.classList.remove('is-resizing');
            }

            resizer.addEventListener('mousedown', onPointerDown);
            resizer.addEventListener('touchstart', onPointerDown, { passive: false });
            resizer.addEventListener('dblclick', (e) => {
                e.preventDefault();
                toggleSidebar();
            });

            window.addEventListener('mousemove', onPointerMove, { passive: false });
            window.addEventListener('touchmove', onPointerMove, { passive: false });

            window.addEventListener('mouseup', onPointerUp);
            window.addEventListener('touchend', onPointerUp);
            window.addEventListener('blur', onPointerUp);
        }

        function checkResponsive() {
            const isNarrow = window.innerWidth <= SHRINK_BREAKPOINT;
            const manualOverride = localStorage.getItem('oly_sidebar_manual');

            if (isNarrow) {
                document.body.classList.add('sidebar-shrunk');
                document.documentElement.classList.add('sidebar-shrunk');
            } else if (!manualOverride || localStorage.getItem('oly_sidebar_shrunk') !== 'true') {
                document.body.classList.remove('sidebar-shrunk');
                document.documentElement.classList.remove('sidebar-shrunk');
                const savedWidth = getSanitizedWidth();
                document.documentElement.style.setProperty('--sidebar-width', savedWidth);
            }
        }

        window.addEventListener('resize', checkResponsive, { passive: true });

        function toggleSidebar() {
            try {
                localStorage.setItem('oly_sidebar_manual', 'true');
            } catch (e) {}
            const isShrunk = document.body.classList.toggle('sidebar-shrunk');
            document.documentElement.classList.toggle('sidebar-shrunk', isShrunk);
            try {
                localStorage.setItem('oly_sidebar_shrunk', isShrunk ? 'true' : 'false');
            } catch (e) {}
            if (!isShrunk) {
                const savedWidth = getSanitizedWidth();
                document.documentElement.style.setProperty('--sidebar-width', savedWidth);
            }
        }

        // Brand logo or dedicated collapse button toggles sidebar
        document.addEventListener('click', (e) => {
            const trigger = e.target.closest('.suite-brand-logo, .suite-sidebar-toggle, #sidebarToggleBtn, #sidebarCollapseBtn');
            if (trigger) {
                const insideSidebar = trigger.closest('.suite-sidebar');
                if (insideSidebar) {
                    e.preventDefault();
                    e.stopPropagation();
                    toggleSidebar();
                }
            }
        });

        // Keyboard shortcut: Ctrl + B or Cmd + B
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key && e.key.toLowerCase() === 'b') {
                if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
                e.preventDefault();
                toggleSidebar();
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSidebarController);
    } else {
        initSidebarController();
    }
})();
