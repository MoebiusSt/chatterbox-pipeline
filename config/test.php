/**
 * Steady Hardpaywall - Server-seitige Cookie-Überprüfung
 * 
 * CSS-Klassen die gesetzt werden:
 * - .steady-checking: Während der Überprüfung (zeigt Loader)
 * - .steady-paying-member: User hat steady-token Cookie (voller Zugang)
 * - .steady-script-available: Kein Cookie, aber Script verfügbar (normale Paywall)
 * - .steady-script-blocked: Kein Cookie und Script blockiert (Warnung anzeigen)
 * 
 * Füge diese Funktionen in die functions.php deines Child-Themes ein
 */

// Check if user is paying Steady member or has admin privileges
function is_steady_paying_member() {
    // WordPress Admins and Editors can always see full content
    if (current_user_can('edit_posts')) {
        return true;
    }
    
    // Debug: Log all cookies for troubleshooting
    if (defined('WP_DEBUG') && WP_DEBUG) {
        error_log('Steady Debug - All cookies: ' . print_r($_COOKIE, true));
    }
    
    // Check for Steady token cookie (try multiple possible names)
    $possible_cookie_names = ['steady-token', 'steady_token', 'steadyToken'];
    
    foreach ($possible_cookie_names as $cookie_name) {
        if (isset($_COOKIE[$cookie_name]) && !empty($_COOKIE[$cookie_name])) {
            if (defined('WP_DEBUG') && WP_DEBUG) {
                error_log('Steady Debug - Found cookie: ' . $cookie_name . ' = ' . $_COOKIE[$cookie_name]);
            }
            return true;
        }
    }
    
    return false;
}

// Content filter for posts/pages
function filter_content_for_steady($content) {
    // Skip filtering in admin, feeds, etc.
    if (is_admin() || is_feed() || !is_main_query()) {
        return $content;
    }
    
    // For paying members: remove paywall marker and show full content
    if (is_steady_paying_member()) {
        // Remove the paywall marker from content
        $content = str_replace('___STEADY_PAYWALL___', '', $content);
        return $content;
    }
    
    // For non-paying members: show teaser + paywall
    return create_teaser_content($content);
}

// Create teaser content with paywall
function create_teaser_content($full_content) {
    // Look for the Steady paywall marker
    $paywall_marker = '___STEADY_PAYWALL___';
    
    // Check if the marker exists in content
    $marker_position = strpos($full_content, $paywall_marker);
    
    if ($marker_position !== false) {
        // Extract everything before the marker as teaser
        $teaser = substr($full_content, 0, $marker_position);
        
        // Clean up any trailing whitespace or incomplete HTML tags
        $teaser = trim($teaser);
        
        // Optional: Close any open HTML tags to prevent broken markup
        $teaser = force_balance_tags($teaser);
    } else {
        // Fallback: if no marker found, show first paragraph or 300 characters
        if (preg_match('/<p[^>]*>(.*?)<\/p>/s', $full_content, $matches)) {
            $teaser = '<p>' . $matches[1] . '</p>';
        } else {
            $teaser = wp_trim_words($full_content, 50, '...');
        }
        
        // Add a note for editors that no marker was found
        if (current_user_can('edit_posts')) {
            $teaser .= '<p><em style="color: #666; font-size: 0.9em;">Hinweis für Redakteure: Kein ___STEADY_PAYWALL___ Marker gefunden. Standard-Teaser wird angezeigt.</em></p>';
        }
    }
    
    // Add paywall overlay
    $paywall_html = '
    <div class="steady-paywall-container">
        <div class="steady-teaser-content">
            ' . $teaser . '
        </div>
        
        <div class="steady-paywall-overlay">
            <div class="steady-paywall-message">
                <h3>Dieser Artikel ist für Steady-Mitglieder</h3>
                <p>Unterstützen Sie uns und erhalten Sie Zugang zu allen Premium-Inhalten.</p>
                <div class="steady-paywall-buttons">
                    <a href="#" class="steady-login-btn" onclick="openSteadyLogin()">Bereits Mitglied? Anmelden</a>
                    <a href="https://steady.page/de/zwiefach-online/about" class="steady-subscribe-btn" target="_blank">Jetzt Mitglied werden</a>
                </div>
            </div>
        </div>
        
        <!-- Steady Widget wird hier geladen -->
        <div id="steady-widget-container"></div>
    </div>';
    
    return $paywall_html;
}

// Apply content filter
add_filter('the_content', 'filter_content_for_steady');

// Add CSS and JavaScript for paywall styling and functionality
function add_steady_hardpaywall_assets() {
    if (is_admin()) {
        return;
    }
    
    ?>
    <style>
        /* Steady Paywall Container - sehr spezifische Selektoren */
        .steady-paywall-container {
            position: relative;
            margin: 20px 0;
            clear: both;
        }
        
        .steady-paywall-container .steady-teaser-content {
            position: relative;
            /* max-height: 200px; */
            /* overflow: hidden; */
            /* mask: linear-gradient(to bottom, black 70%, transparent 100%); */
            /* -webkit-mask: linear-gradient(to bottom, black 70%, transparent 100%); */
        }
        
        .steady-paywall-container .steady-paywall-overlay {
            background: linear-gradient(
                to bottom, 
                rgba(255,255,255,0) 0%, 
                rgba(255,255,255,0.8) 50%, 
                rgba(255,255,255,1) 100%
            );
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 150px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding: 20px;
        }
        
        .steady-paywall-container .steady-paywall-message {
            background: white;
            border: 2px solid #291e38;
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 500px;
            margin: 0;
            z-index: 999;
        }
        
        .steady-paywall-container .steady-paywall-message h3 {
            margin: 0 0 15px 0;
            padding: 0;
            color: #291e38;
            font-size: 1.3em;
            font-weight: bold;
            line-height: 1.3;
        }
        
        .steady-paywall-container .steady-paywall-message p {
            margin: 0 0 10px 0;
            padding: 0;
            color: #333;
            font-size: 1em;
            line-height: 1.5;
        }
        
        .steady-paywall-container .steady-paywall-buttons {
            margin: 20px 0 0 0;
            padding: 0;
        }
        
        .steady-paywall-container .steady-paywall-buttons a {
            display: inline-block;
            padding: 12px 24px;
            margin: 5px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            font-size: 14px;
            line-height: 1;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .steady-paywall-container .steady-login-btn {
            background: #f0f0f0;
            color: #333 !important;
            border: 2px solid #ddd;
        }
        
        .steady-paywall-container .steady-login-btn:hover {
            background: #e0e0e0;
            border-color: #ccc;
            color: #333 !important;
        }
        
        .steady-paywall-container .steady-subscribe-btn {
            background: #291e38;
            color: white !important;
            border: 2px solid #291e38;
        }
        
        .steady-paywall-container .steady-subscribe-btn:hover {
            background: #1d1527;
            border-color: #1d1527;
            color: white !important;
        }
        
        /* Body class specific styles - nur wenn Steady-Klassen gesetzt sind */
        body.steady-paying-member .steady-paywall-container {
            display: none;
        }
        
        body.steady-paying-member .steady-full-content {
            display: block;
        }
        
        /* Scriptblocker warning - sehr spezifisch */
        .steady-script-blocked-warning {
            display: none;
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            margin: 20px auto;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            text-align: center;
            max-width: 600px;
            box-sizing: border-box;
        }
        
        .steady-script-blocked-warning h3 {
            margin: 0 0 15px 0;
            padding: 0;
            color: #721c24;
            font-size: 1.2em;
        }
        
        .steady-script-blocked-warning p {
            margin: 0 0 10px 0;
            padding: 0;
            color: #721c24;
        }
        
        /* Only show warning when script is blocked */
        body.steady-script-blocked .steady-paywall-container {
            display: none;
        }
        
        body.steady-script-blocked .steady-script-blocked-warning {
            display: block;
        }
        
        /* Responsive design for paywall */
        @media (max-width: 600px) {
            .steady-paywall-container .steady-paywall-message {
                margin: 0 10px;
                padding: 20px 15px;
            }
            
            .steady-paywall-container .steady-paywall-buttons a {
                display: block;
                margin: 10px 0;
                text-align: center;
            }
        }
    </style>
    
    <script type="text/javascript">
        // Client-side enhancement: Check for scriptblockers
        function checkForScriptBlockers() {
            // Double-check: Are we a paying member client-side too?
            function getCookie(name) {
                var value = "; " + document.cookie;
                var parts = value.split("; " + name + "=");
                if (parts.length == 2) {
                    return parts.pop().split(";").shift();
                }
                return null;
            }
            
            var steadyToken = getCookie('steady-token');
            if (steadyToken) {
                console.log('Steady: Paying member cookie found client-side - no script check needed');
                return;
            }
            
            // Only run this check for non-paying members (server-side check)
            <?php if (!is_steady_paying_member()): ?>
            
            var hasScriptAccess = false;
            
            // Check if Steady script loaded
            setTimeout(function() {
                if (typeof window.SteadyWidget !== 'undefined' || 
                    typeof window.steady !== 'undefined' ||
                    document.querySelector('script[src*="steadyhq.com"]')) {
                    hasScriptAccess = true;
                }
                
                if (!hasScriptAccess) {
                    // Show scriptblocker warning instead of paywall
                    document.body.classList.add('steady-script-blocked');
                    console.log('Steady: Script blocked - showing warning');
                } else {
                    console.log('Steady: Script available - showing normal paywall');
                }
            }, 2000);
            
            <?php else: ?>
            // User is paying member - no need to check for script blockers
            console.log('Steady: Paying member detected server-side - skipping script blocker check');
            <?php endif; ?>
        }
        }
        
        // Function to open Steady login (customize as needed)
        function openSteadyLogin() {
            // This depends on how Steady implements login
            // You might need to customize this based on Steady's API
            if (typeof window.SteadyWidget !== 'undefined') {
                // Use Steady's login function if available
                window.SteadyWidget.openLogin();
            } else {
                // Fallback: redirect to Steady login page
                window.open('https://steady.page/de/log_in?publication=zwiefach-online', '_blank');
            }
        }
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', checkForScriptBlockers);
        } else {
            checkForScriptBlockers();
        }
    </script>
    
    <?php
    // Add body class for paying members
    if (is_steady_paying_member()) {
        echo '<script>document.body.classList.add("steady-paying-member");</script>';
    }
}
add_action('wp_head', 'add_steady_hardpaywall_assets');

// Add the scriptblocker warning HTML to footer
function add_scriptblocker_warning() {
    if (is_admin() || is_steady_paying_member()) {
        return;
    }
    
    echo '<div class="steady-script-blocked-warning">
        <h3>Script-Blocker erkannt</h3>
        <p>Bitte deaktivieren Sie Ihren Werbeblocker oder Script-Blocker für diese Website, um alle Inhalte anzuzeigen.</p>
        <p>Laden Sie die Seite nach der Deaktivierung neu.</p>
    </div>';
}
add_action('wp_footer', 'add_scriptblocker_warning');

// Ensure Steady script is loaded (if not already present)
function ensure_steady_script() {
    if (is_admin() || is_steady_paying_member()) {
        return;
    }
    
    ?>
    <script type="text/javascript" src="https://steadyhq.com/widget_loader/3db015a4-dc44-4836-9ad2-a3a9e1221afa"></script>
    <?php
}
add_action('wp_head', 'ensure_steady_script');

/**
 * Optional: Redirect function for protected pages
 * Use this if you want to completely redirect non-paying users
 */
function redirect_non_paying_users() {
    // Only on specific pages/posts (customize as needed)
    if (is_admin() || !is_singular()) {
        return;
    }
    
    // Check if this post requires Steady membership
    $requires_membership = get_post_meta(get_the_ID(), 'requires_steady_membership', true);
    
    if ($requires_membership && !is_steady_paying_member()) {
        // Redirect to subscription page
        wp_redirect('https://steady.page/de/zwiefach-online/about');
        exit;
    }
}
// Uncomment to enable redirects:
// add_action('template_redirect', 'redirect_non_paying_users');

/**
 * Helper function to mark posts as requiring membership
 * Add this to your post edit screen or use programmatically
 */
function mark_post_as_premium($post_id) {
    update_post_meta($post_id, 'requires_steady_membership', true);
}

/**
 * Admin interface to mark posts/pages as premium (optional)
 */
function add_steady_meta_box() {
    // Add meta box for both posts and pages
    $post_types = array('post', 'page');
    
    foreach ($post_types as $post_type) {
        add_meta_box(
            'steady-membership',
            'Steady Mitgliedschaft',
            'steady_meta_box_callback',
            $post_type,
            'side',
            'high'
        );
    }
}

function steady_meta_box_callback($post) {
    $requires_membership = get_post_meta($post->ID, 'requires_steady_membership', true);
    
    wp_nonce_field('steady_meta_box_nonce', 'steady_meta_box_nonce');
    
    echo '<label style="margin-bottom: 15px; display: block;">
        <input type="checkbox" name="requires_steady_membership" value="1" ' . checked($requires_membership, true, false) . '> 
        Dieser Inhalt erfordert eine Steady-Mitgliedschaft
    </label>';
    
    echo '<div style="background: #f9f9f9; padding: 10px; border-left: 4px solid #007cba; margin-top: 10px;">
        <strong>Paywall-Position festlegen:</strong><br>
        Fügen Sie <code>___STEADY_PAYWALL___</code> an der Stelle in Ihren Inhalt ein, wo die Paywall beginnen soll.<br>
        <small style="color: #666;">Alles nach diesem Marker wird nur für Steady-Mitglieder angezeigt.</small>
    </div>';
}

function save_steady_meta_box($post_id) {
    // Check nonce for security
    if (!isset($_POST['steady_meta_box_nonce']) || !wp_verify_nonce($_POST['steady_meta_box_nonce'], 'steady_meta_box_nonce')) {
        return;
    }
    
    // Check if user can edit this post/page
    if (!current_user_can('edit_post', $post_id)) {
        return;
    }
    
    // Don't save during autosave
    if (defined('DOING_AUTOSAVE') && DOING_AUTOSAVE) {
        return;
    }
    
    // Save the membership requirement
    if (isset($_POST['requires_steady_membership'])) {
        update_post_meta($post_id, 'requires_steady_membership', true);
    } else {
        delete_post_meta($post_id, 'requires_steady_membership');
    }
}

// Enable admin interface:
add_action('add_meta_boxes', 'add_steady_meta_box');
add_action('save_post', 'save_steady_meta_box');

/**
 * Add Quick Insert Button for Paywall Marker (optional)
 */
function add_paywall_marker_button() {
    global $post_type;
    if (in_array($post_type, array('post', 'page'))) {
        echo '<script>
        jQuery(document).ready(function($) {
            // Add button to WordPress editor
            if (typeof QTags !== "undefined") {
                QTags.addButton("steady_paywall", "Steady Paywall", "___STEADY_PAYWALL___", "", "", "Steady Paywall Marker einfügen");
            }
            
            // For Classic Editor: add button above textarea
            $("#content").before(\'<div style="margin-bottom: 10px;"><button type="button" class="button" onclick="insertSteadyMarker()">📄 Steady Paywall hier einfügen</button></div>\');
        });
        
        function insertSteadyMarker() {
            var editor = window.tinymce && window.tinymce.get("content");
            if (editor && !editor.isHidden()) {
                // Visual editor
                editor.insertContent("___STEADY_PAYWALL___");
            } else {
                // Text editor
                var textarea = document.getElementById("content");
                if (textarea) {
                    var cursorPos = textarea.selectionStart;
                    var textBefore = textarea.value.substring(0, cursorPos);
                    var textAfter = textarea.value.substring(cursorPos);
                    textarea.value = textBefore + "___STEADY_PAYWALL___" + textAfter;
                    textarea.setSelectionRange(cursorPos + 19, cursorPos + 19);
                    textarea.focus();
                }
            }
        }
        </script>';
    }
}

// Enable quick insert button:
add_action('admin_footer-post.php', 'add_paywall_marker_button');
add_action('admin_footer-post-new.php', 'add_paywall_marker_button');