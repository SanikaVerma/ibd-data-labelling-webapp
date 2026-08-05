# ============================================================================
# PATIENT TIMELINE VIEWER - MAIN APPLICATION INTERFACE
# ============================================================================
# 
# This is the main entry point for the Patient Timeline Viewer application.
# It creates a Gradio web interface for doctors to view patient medical timelines
# and add/manage disease flare periods.
#
# The application supports:
# - Multiple patient access groups (command-line controlled)
# - Timeline visualization with event filtering
# - Monthly flare labelling system
# - Data export capabilities
#
# Usage: python main.py [group-a|group-b|group-c|custom|admin|dev] [options]
# ============================================================================

import gradio as gr
import argparse
import os
import sys
from patient_timeline_webapp import PatientTimelineApp
from monthly_labelling import add_monthly_labelling_methods
from timeline_visualization import create_main_timeline, create_monthly_timeline

# Global app instance - will be initialized with command line arguments
app = None

def parse_arguments():
    """
    Parse command line arguments to determine patient access permissions.
    
    This function sets up different patient groups to control which patients
    each user/team can access. This is useful for:
    - Research teams working on specific cohorts  
    - Clinical teams with different responsibilities
    - Limiting access for security/privacy reasons
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Patient Timeline Viewer with Patient Group Access Control')
    
    # Create subcommands for different patient access groups
    subparsers = parser.add_subparsers(dest='command', help='Available patient groups')
    
    # GROUP A: Patients 1-100 (e.g., for Research Team A)
    group_a = subparsers.add_parser('group-a', help='Access patients 1-100')
    group_a.add_argument('--start', type=int, default=1, help='Start patient ID (default: 1)')
    group_a.add_argument('--end', type=int, default=100, help='End patient ID (default: 100)')
    group_a.add_argument('--rmt23345-dir', type=str, default=None, dest='rmt23345_dir',
                         help='Directory containing cleaned RMT23345_*.csv files')

    # GROUP B: Patients 101-200 (e.g., for Research Team B)
    group_b = subparsers.add_parser('group-b', help='Access patients 101-200')
    group_b.add_argument('--start', type=int, default=101, help='Start patient ID (default: 101)')
    group_b.add_argument('--end', type=int, default=200, help='End patient ID (default: 200)')
    group_b.add_argument('--rmt23345-dir', type=str, default=None, dest='rmt23345_dir',
                         help='Directory containing cleaned RMT23345_*.csv files')

    # GROUP C: Patients 201-300 (e.g., for Clinical Team)
    group_c = subparsers.add_parser('group-c', help='Access patients 201-300')
    group_c.add_argument('--start', type=int, default=201, help='Start patient ID (default: 201)')
    group_c.add_argument('--end', type=int, default=300, help='End patient ID (default: 300)')
    group_c.add_argument('--rmt23345-dir', type=str, default=None, dest='rmt23345_dir',
                         help='Directory containing cleaned RMT23345_*.csv files')

    # CUSTOM RANGE: For specific research projects or temporary access
    custom = subparsers.add_parser('custom', help='Access custom patient range')
    custom.add_argument('--start', type=int, required=True, help='Start patient ID')
    custom.add_argument('--end', type=int, required=True, help='End patient ID')
    custom.add_argument('--name', type=str, default='Custom', help='Group name for display')
    custom.add_argument('--rmt23345-dir', type=str, default=None, dest='rmt23345_dir',
                        help='Directory containing cleaned RMT23345_*.csv files')

    # ADMIN ACCESS: All patients (for administrators/supervisors)
    admin = subparsers.add_parser('admin', help='Access all patients (admin mode)')
    admin.add_argument('--rmt23345-dir', type=str, default=None, dest='rmt23345_dir',
                       help='Directory containing cleaned RMT23345_*.csv files')
    
    return parser.parse_args()

def create_app_with_args():
    """
    Create PatientTimelineApp instance based on command line arguments.
    
    This function interprets the parsed command line arguments and creates
    the appropriate app instance with the correct patient access permissions.
    
    Returns:
        PatientTimelineApp: Configured app instance
    """
    args = parse_arguments()
    
    # Create app instance based on which command was used
    if args.command == 'group-a':
        pr = None if args.rmt23345_dir else (args.start, args.end)
        return PatientTimelineApp(patient_range=pr, group_name="Group A",
                                  rmt23345_data_dir=args.rmt23345_dir)
    elif args.command == 'group-b':
        pr = None if args.rmt23345_dir else (args.start, args.end)
        return PatientTimelineApp(patient_range=pr, group_name="Group B",
                                  rmt23345_data_dir=args.rmt23345_dir)
    elif args.command == 'group-c':
        pr = None if args.rmt23345_dir else (args.start, args.end)
        return PatientTimelineApp(patient_range=pr, group_name="Group C",
                                  rmt23345_data_dir=args.rmt23345_dir)
    elif args.command == 'custom':
        pr = None if args.rmt23345_dir else (args.start, args.end)
        return PatientTimelineApp(patient_range=pr, group_name=args.name,
                                  rmt23345_data_dir=args.rmt23345_dir)
    elif args.command == 'admin':
        return PatientTimelineApp(patient_range=None, group_name="Admin (All Patients)",
                                  rmt23345_data_dir=args.rmt23345_dir)
    else:
        # No valid command provided - show help and exit
        print("No command specified. Available commands:")
        print("  python main.py group-a    # Access patients 1-100")
        print("  python main.py group-b    # Access patients 101-200") 
        print("  python main.py group-c    # Access patients 201-300")
        print("  python main.py custom --start 1 --end 50 --name 'My Group'")
        print("  python main.py admin      # Access all patients")
        sys.exit(1)

def create_interface():
    """
    Create the main Gradio web interface for the Patient Timeline Viewer.
    
    This function builds the complete user interface with three main tabs:
    1. Data Overview - Data loading status and export capabilities
    2. Timeline Viewer - Main timeline visualization with filtering
    3. Labelling Mode - Monthly flare labelling interface
    4. User Guide - Documentation and help
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    
    # Create the main Gradio interface with custom theme
    with gr.Blocks(
        title=f"Patient Timeline Viewer - {app.group_name}", 
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Inter"), "Arial", "Roboto"], 
            primary_hue="blue"  # Blue theme for medical application
        )
    ) as demo:
        
        # ====================================================================
        # CUSTOM CSS STYLING
        # ====================================================================
        # This CSS provides a clean, professional look for the medical interface
        gr.HTML("""
            <style>
            /* Section title styling with blue underline */
            .section-title {
                font-size: 16px !important;
                font-weight: 600 !important;
                color: #1f2937 !important;
                margin-bottom: 12px !important;
                padding-bottom: 6px !important;
                border-bottom: 2px solid #3b82f6 !important;
                display: block !important;
            }
            
            /* Clean control column - no boxes, just spacing */
            .clean-control-column {
                padding-right: 20px !important;
            }
            
            .clean-control-column > * {
                margin-bottom: 12px !important;
            }
            
            .clean-control-column .section-title {
                margin-top: 10px !important;
                margin-bottom: 8px !important;
            }
            iu
            .clean-control-column .section-title:first-child {
                margin-top: 0 !important;
            }
            
            /* Clean month dropdown styling */
            .clean-month-dropdown {
                margin-bottom: 12px !important;
            }
            
            .clean-month-dropdown .gr-form {
                background: white !important;
                border: 1px solid #d1d5db !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
            }
            
            .clean-month-dropdown .gr-form:focus-within {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            }
            
            /* Month navigation row */
            .month-nav-row {
                display: flex !important;
                gap: 8px !important;
                margin-bottom: 16px !important;
            }
            
            .month-nav-button {
                flex: 1 !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
                font-size: 14px !important;
                font-weight: 500 !important;
            }
            
            .month-nav-button:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }
            
            /* Timeline navigation styling */
            .timeline-nav {
                margin-bottom: 12px !important;
                padding: 8px 0 !important;
                display: flex !important;
                align-items: center !important;
                gap: 12px !important;
            }
            
            .timeline-nav button {
                border-radius: 8px !important;
                padding: 8px 16px !important;
                font-size: 14px !important;
                font-weight: 500 !important;
            }
            
            /* Better alignment for labelling top row */
            .labelling-top-row {
                align-items: flex-start !important;
                margin-bottom: 25px !important;
            }
            
            .labelling-top-row .section-title {
                margin-top: 0 !important;
            }
            
            /* Better button styling */
            .gradio-button {
                border-radius: 8px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            
            .gradio-button:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }
            
            /* Improve form elements */
            .gr-form {
                border-radius: 8px !important;
                border: 1px solid #d1d5db !important;
                transition: border-color 0.2s ease !important;
            }
            
            .gr-form:focus-within {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            }
            
            /* Better spacing for radio buttons */
            .gr-radio {
                margin-bottom: 12px !important;
            }
            
            /* Improve textbox styling */
            .gr-textbox {
                border-radius: 8px !important;
            }
            
            /* Clean up dropdown appearance */
            .gr-dropdown {
                border-radius: 8px !important;
            }
            
            /* Better plot container */
            .gr-plot {
                border-radius: 10px !important;
                border: 1px solid #e5e7eb !important;
                overflow: hidden !important;
            }
            
            /* Remove default gradio group styling */
            .gradio-group {
                border: none !important;
                background: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            
            /* Clean spacing for form elements */
            .gr-box {
                margin-bottom: 12px !important;
            }
            
            /* Better row spacing */
            .gr-row {
                margin-bottom: 8px !important;
            }
            
            /* Ensure consistent spacing between elements */
            .clean-control-column .gr-dropdown,
            .clean-control-column .gr-radio,
            .clean-control-column .gr-textbox {
                margin-bottom: 12px !important;
            }
            
            /* Button row spacing */
            .clean-control-column .gr-row {
                margin-bottom: 12px !important;
            }
            </style>
            """
        )
        
        # ====================================================================
        # MAIN HEADER
        # ====================================================================
        gr.Markdown(f"# Patient Timeline Viewer - {app.group_name}")
        gr.Markdown("A comprehensive tool for visualizing patient timelines and managing flare periods.")
        
        # ====================================================================
        # TAB 1: DATA OVERVIEW
        # ====================================================================
        # This tab shows current data status and provides export functionality
        with gr.Tab("Data Overview"):
            gr.Markdown("### Current Data Status")

            with gr.Row():
                # Left column: Data information and reload functionality
                with gr.Column():
                    gr.Markdown("**Data Information**")
                    # Shows current data loading status (patient count, date range, etc.)
                    data_status = gr.Textbox(label="Status", value=app.get_data_status(), interactive=False)
                    # Button to reload data from the file system
                    reload_data_btn = gr.Button("Reload Data", variant="primary")

                # Right column: Data export functionality
                with gr.Column():
                    gr.Markdown("**Export Current Data**")
                    # Radio buttons to choose export format
                    export_format = gr.Radio(["CSV", "Excel"], label="Export format", value="CSV")
                    # Button to trigger data export
                    export_btn = gr.Button("Export Data")
                    # Status message for export operations
                    export_status = gr.Textbox(label="Export Status", value="", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Upload Study Config")
            gr.Markdown("Upload your `study_config.yaml` with paths to your data files on this server.")

            with gr.Row():
                upload_config = gr.UploadButton("↑ Upload study_config.yaml", file_types=[".yaml", ".yml"], size="sm")
                config_name   = gr.Textbox(value="no file", show_label=False, interactive=False, max_lines=1, lines=1)

            with gr.Row():
                load_config_btn = gr.Button("Load Config & Data", variant="primary")
                config_status   = gr.Textbox(label="Status", value="", interactive=False, lines=1)
        
        # ====================================================================
        # TAB 2: TIMELINE VIEWER
        # ====================================================================
        # Main timeline visualization tab for viewing patient medical events
        with gr.Tab("Timeline Viewer"):
            with gr.Column():
                # Top row: Patient selection and filtering controls
                with gr.Row():
                    # Left column: Patient selection
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("### Patient Selection")
                        # Dropdown populated with available patient IDs based on access permissions
                        patient_dropdown = gr.Dropdown(
                            label="Patient ID", 
                            choices=app.get_patient_choices(),
                            value=app.get_patient_choices()[0] if app.get_patient_choices() else None,
                            interactive=True
                        )
                        # Button to manually load timeline (though auto-loading is also enabled)
                        load_timeline_btn = gr.Button("Load Timeline", variant="primary")
                    
                    # Right column: Chart controls and filters
                    with gr.Column(scale=5):
                        gr.Markdown("### Timeline Chart")
                        # Status message for chart operations
                        chart_status = gr.Textbox(label="Chart Status", value="No chart loaded", interactive=False)

                # Timeline (client-side Plotly.js) alongside the XAI importance list
                with gr.Row():
                    with gr.Column(scale=3):
                        # IBD filter buttons live inside the HTML component (no Python round-trip)
                        timeline_plot = gr.HTML(
                            value="<p style='color:#6b7280;padding:20px;text-align:center;'>"
                                  "Select a patient and click Load Timeline to view the chart.</p>"
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### Most Important Items (XAI mock-up)")
                        xai_list = gr.HTML(
                            value="<p style='color:#6b7280;padding:12px;'>"
                                  "Select a patient to see importance scores.</p>"
                        )
                        gr.Markdown("#### Clinical note — words by importance")
                        note_selector = gr.Dropdown(
                            label="Select a note",
                            choices=[], value=None, interactive=True
                        )
                        note_gradient = gr.Plot()

                # Chart information panel - shows patient stats and flare summaries
                chart_info = gr.Textbox(label="Chart Information", value="No patient data loaded",
                                      lines=3, interactive=False, max_lines=4)

                # Readmission probability curve (per patient), below the timeline.
                gr.Markdown("### Readmission probability")
                readmission_curve = gr.Plot()
                readmission_table = gr.HTML(
                    value="<p style='color:#6b7280;padding:8px;'>Select a patient.</p>"
                )
        
        # ====================================================================
        # TAB 3: LABELLING MODE
        # ====================================================================
        # Monthly flare labelling interface for medical professionals
        with gr.Tab("Labelling Mode"):
            # Top row: Patient selection and status (aligned headers for clean look)
            with gr.Row(elem_classes="labelling-top-row"):
                # Left: Patient selection for labelling
                with gr.Column(scale=1):
                    gr.HTML("<h3 class='section-title'>Patient Selection</h3>")
                    # Separate patient dropdown for labelling mode
                    patient_dropdown_label = gr.Dropdown(
                        label="Patient ID", 
                        choices=app.get_patient_choices(),
                        value=app.get_patient_choices()[0] if app.get_patient_choices() else None,
                        interactive=True
                    )
                    # Load patient specifically for labelling (loads all events, not filtered)
                    load_labelling_btn = gr.Button("Load Patient", variant="primary", size="lg")
                
                # Right: Chart status for labelling operations
                with gr.Column(scale=1):
                    gr.HTML("<h3 class='section-title'>Chart Status</h3>")
                    # Status updates for labelling operations
                    label_chart_status = gr.Textbox(
                        label="Status",
                        value="Select and load a patient to begin labelling",
                        interactive=False
                    )
            
            # Main content area: Controls and visualization
            with gr.Row():
                # ============================================================
                # LEFT COLUMN: LABELLING CONTROLS (Clean single column design)
                # ============================================================
                with gr.Column(scale=1, min_width=320, elem_classes="clean-control-column"):
                    
                    # --- MONTH SELECTION SECTION ---
                    gr.HTML("<h3 class='section-title'>Generating Label For</h3>")
                    # Dropdown showing only months that have medical events
                    month_dropdown = gr.Dropdown(
                        label="Select Month",
                        choices=[],  # Populated after patient is loaded
                        interactive=True,
                        elem_classes="clean-month-dropdown"
                    )
                    
                    # Month navigation buttons (separate row for clean layout)
                    with gr.Row(elem_classes="month-nav-row"):
                        month_back_btn = gr.Button("◀ Previous Month", 
                                                 size="sm", 
                                                 variant="secondary", 
                                                 elem_classes="month-nav-button")
                        month_forward_btn = gr.Button("Next Month ▶", 
                                                    size="sm", 
                                                    variant="secondary", 
                                                    elem_classes="month-nav-button")
                    
                    # --- FLARE LABELLING SECTION ---
                    gr.HTML("<h3 class='section-title'>Flare Management</h3>")
                    
                    # Main flare evidence question (Yes/No)
                    flare_evidence = gr.Radio(
                        choices=["Yes", "No"],
                        label="Evidence of Flare",
                        value="No",  # Default to No to avoid accidental positive labels
                        interactive=True
                    )
                    
                    # Category selection (required if evidence = Yes)
                    # These categories correspond to different types of medical events
                    category_dropdown_label = gr.Dropdown(
                        label="Category (Required if Yes)",
                        choices=[
                            "Ambulatory Visit",    # Outpatient visits
                            "Lab Test",           # Laboratory tests
                            "Prescription",       # Medication prescriptions
                            "Physician Claim",    # Insurance claims for physician services
                            "Hospital Admission", # Inpatient hospital stays
                            "Imaging"            # Medical imaging (X-ray, CT, etc.)
                        ],
                        value=[],
                        multiselect=True,  # Allow multiple categories per flare
                        interactive=True
                    )
                    
                    # Optional reason/notes field
                    reason_input_label = gr.Textbox(
                        label="Reason (Optional)", 
                        placeholder="Enter reason for flare",
                        lines=2
                    )

                    # Save/Clear buttons for current month
                    with gr.Row():
                        save_label_btn = gr.Button("Save Label", variant="primary", scale=1)
                        clear_label_btn = gr.Button("Clear Label", variant="secondary", scale=1)

                    # --- EDIT EXISTING LABELS SECTION ---
                    gr.HTML("<h3 class='section-title'>Edit Existing Label</h3>")
                    
                    # Dropdown showing months that already have labels
                    edit_month_dropdown = gr.Dropdown(
                        label="Select Month to Edit",
                        choices=[],  # Populated with months that have existing labels
                        interactive=True
                    )

                    # Edit/Delete buttons for existing labels
                    with gr.Row():
                        load_edit_btn = gr.Button("Load for Edit", variant="secondary", scale=1)
                        delete_label_btn = gr.Button("Delete Label", variant="stop", scale=1)

                    # --- SAVED LABELS SUMMARY ---
                    gr.HTML("<h3 class='section-title'>Saved Labels</h3>")
                    # Text area showing all saved labels for current patient
                    labels_info = gr.Textbox(
                        label="Labels Summary",
                        value="No labels saved yet",
                        interactive=False,
                        lines=8,
                        max_lines=12
                    )

                # ============================================================
                # RIGHT COLUMN: TIMELINE VISUALIZATION (3/4 width)
                # ============================================================
                with gr.Column(scale=3):
                    # Timeline navigation buttons at top of chart
                    with gr.Row(elem_classes="timeline-nav"):
                        # These buttons allow viewing different months while keeping
                        # the labelling target month separate
                        view_back_btn_visible = gr.Button("◀ Previous Month", 
                                                        size="sm", 
                                                        variant="secondary")
                        gr.HTML("<div style='flex: 1;'></div>")  # Centered spacer
                        view_forward_btn_visible = gr.Button("Next Month ▶", 
                                                           size="sm", 
                                                           variant="secondary")
                    
                    # Main monthly timeline plot
                    # Shows one month at a time with detailed event visualization
                    monthly_timeline_plot = gr.Plot()
                    
            # Hidden element needed for event handlers (not visible to user)
            current_view_info_visible = gr.Textbox(
                label="Current View",
                value="",
                interactive=False,
                visible=False  # Used internally for state management
            )
        
        # ====================================================================
        # TAB 4: USER GUIDE
        # ====================================================================
        # Comprehensive documentation for users
        with gr.Tab("User Guide"):
            gr.Markdown("""
            ## Patient Timeline Viewer - User Guide

                ### 1. Data Loading & Access Control
                - **Command-Based Access**: Different commands provide access to different patient groups
                - **Available Commands**:
                - `uv run python main.py group-a` - Access patients 1-100
                - `uv run python main.py group-b` - Access patients 101-200  
                - `uv run python main.py group-c` - Access patients 201-300
                - `uv run python main.py custom --start X --end Y --name "Group Name"` - Custom range
                - `uv run python main.py admin` - Access all patients (admin mode)
                - **Save Directories**: Group-specific folders (`groupa_saved_flares/`, `groupb_saved_flares/`, etc.)
                - **Reload Data**: Use the "Reload Data" button to refresh data from the directory
                - **Export Current Data**: Save the currently loaded data to CSV or Excel format

                ### 2. Running on Remote Server
                **Terminal 1 - Start Application**:
                ```bash
                uv run python main.py admin --rmt23345-dir /path/to/data

                **Terminal 2 - SSH Tunnel**:
                ```bash
                ssh -L 7860:localhost:7860 username@server-hostname

                Access: Open browser to http://localhost:7860

                ###3. Timeline Viewer & Event Filtering
                - **Patient Selection**: Choose a patient from the dropdown (automatically populated from your assigned patient group)
                - **IBD Event Filtering**: Use the radio buttons above the chart to filter events:
                - **"All Events"** - Show all medical events (default)
                - **"IBD Related Only"** - Show only IBD-related events (`ibd_related = True`)
                - **"Non-IBD Related Only"** - Show only non-IBD events (`ibd_related = False`)
                - **Real-time Filtering**: Event filter changes apply immediately without reloading
                - **Interactive Chart**: Hover over events for detailed information, zoom and pan as needed
                - **Cross-Tab Flares**: Monthly flares created in Labelling Mode appear as orange highlights with star markers

                ###4. Labelling Mode (Monthly Flare Labeling)
                - **Patient Selection**: Choose a patient for monthly labeling
                - **Month Navigation**: Select specific months using dropdown and navigation buttons
                - **Monthly View**: Timeline shows selected month with navigation to view adjacent months
                - **Evidence Classification**: Mark each month as having flare evidence (Yes/No)
                - **Category Selection**: If evidence exists, select relevant medical event categories
                - **Reason Documentation**: Optional text field for additional details
                - **Efficient Navigation**: Dropdown only shows months with events for faster labeling
                - **Auto-Reset Form**: Form clears automatically when switching months

                ###5. Monthly Timeline Features
                - **Focused View**: Shows one month at a time with context from adjacent periods
                - **Navigation Controls**: Move view forward/backward while maintaining labeling target
                - **Visual Highlighting**: Selected month highlighted with blue background
                - **Persistent Labels**: All monthly labels saved automatically per patient group directories
                - **Flare Highlighting**: Months with Evidence=Yes show red background
                - **Enhanced JSON Format**: Labels include metadata, timestamps, and readable category names
                - **Rich Hover Information**: All flare details available on hover in both tabs
                - **Cross-Tab Synchronization**: Monthly flares automatically appear in Timeline Viewer

                ###6. Lab Test Visualization
                - **Multiple Same-Day Tests**:  When multiple lab tests occur on the same day, they appear as separate adjacent bars with 2-hour time offsets
                - **Visual Separation**: Each lab test maintains its original duration but gets a small time offset for visibility
                - **Individual Hover Data**: Each lab test bar shows its own specific details in the hover tooltip

                ###7. Advanced Features
                - **Command-Line Access Control**: Different teams can access different patient cohorts
                - **Automatic Data Loading**: No need to manually upload files - data is loaded from the specified directory
                Group-Specific Save Directories: Labels saved to separate folders per access group
                - **Rich Hover Information**: View detailed event information with smart text wrapping for long descriptions
                - **Colour-Coded Events**: Different event types (lab tests, prescriptions, visits, etc.) have distinct colours, and always in same Y-axis order
                - **Export Capabilities**: Save charts as images using Plotly's built-in tools
                - **Real-time Chart Updates**: All filtering and flare changes update the chart immediately

                ###8. Data Format Requirements
                The app expects CSV files with "events" in the filename containing:
            
                **Required columns:**
                - `patient_id`: Unique identifier for each patient (integer)
                - `start_date`: Event start date (YYYY-MM-DD format)
                - `end_date`: Event end date (YYYY-MM-DD format)  
                - `event_type`: Type of medical event (string)
                - `ibd_related`: Boolean flag indicating if event is IBD-related (True/False)
                
                **Optional columns:**
                - `event_info`: JSON string with additional details (displayed in hover tooltips)
                - `source_dataset`: Data source identifier (e.g., "CLAIMS", "LAB", "DAD")
                - Any additional columns will be displayed in hover information if part of `event_info`

                ###9. Command Line Examples
                ```bash
                # Research team accessing early patients
                uv run python main.py group-a

                # Clinical team accessing different cohort
                uv run python main.py group-b

                # Custom analysis group
                uv run python main.py custom --start 150 --end 250 --name "Cohort Study"

                # Administrator viewing all patients
                uv run python main.py admin

                ###10. Troubleshooting
                - **No patients visible**: Check that your command provides access to patients in the data range
                - **Missing events**: Verify the IBD filter setting matches what you want to see
                - **Overlapping lab tests**: The app automatically separates same-day lab tests - if still overlapping, try refreshing the chart
                - **Flare not saving**: Ensure dates are in YYYY-MM-DD format and end date is after start date, check group directory permissions (groupa_saved_flares/, etc.)
                - **Data not loading**: Check that the data directory exists and contains CSV files with "events" in the filename
                - **Monthly flares not showing**: Ensure you're viewing the correct patient and the timeline is refreshed
                - **Permission denied**: Ensure group directories have proper permissions (chmod 2775)
                - **SSH tunnel not working**: Verify both terminals are running and port numbers match
                """)
        
        # ====================================================================
        # EVENT HANDLERS - DATA OVERVIEW TAB
        # ====================================================================
        
        # Reload data button: Refreshes data from file system and updates UI
        reload_data_btn.click(
            app.reload_data,  # App method that reloads data from directory
            outputs=[data_status, patient_dropdown, chart_info]  # Update these UI elements
        )
        
        # Export data button: Saves current data to CSV or Excel
        export_btn.click(
            app.export_data,  # App method that handles data export
            inputs=[export_format],  # Use selected format (CSV/Excel)
            outputs=[export_status]  # Show export status message
        )

        # Config YAML upload: reads paths from uploaded YAML and loads data directly
        load_config_btn.click(
            app.load_config_file,
            inputs=[upload_config],
            outputs=[config_status, data_status, patient_dropdown, chart_info]
        )

        def _fname(f):
            if f is None:
                return "no file"
            p = f if isinstance(f, str) else f.name
            return os.path.basename(p)

        upload_config.upload(_fname, inputs=upload_config, outputs=config_name)

        # ====================================================================
        # EVENT HANDLERS - TIMELINE VIEWER TAB
        # ====================================================================
        
        # Refresh the note dropdown when the patient changes; always (re)draw the
        # gradient bar — empty (with a message) until a note is selected.
        def _refresh_notes(patient_id):
            return (gr.update(choices=app.note_choices(patient_id), value=None),
                    app.get_note_gradient_fig(patient_id, None))

        # Redraw the readmission curve + table when the patient changes.
        def _refresh_readmission(patient_id):
            return (app.get_readmission_fig(patient_id),
                    app.get_readmission_table_html(patient_id))

        # Load timeline button — returns HTML/JS with filter buttons embedded
        load_timeline_btn.click(
            app.load_patient_timeline_html,
            inputs=[patient_dropdown],
            outputs=[timeline_plot, chart_status, chart_info, xai_list]
        ).then(
            _refresh_notes, inputs=[patient_dropdown], outputs=[note_selector, note_gradient]
        ).then(
            _refresh_readmission, inputs=[patient_dropdown],
            outputs=[readmission_curve, readmission_table]
        )

        # Auto-load when patient dropdown changes
        patient_dropdown.change(
            app.load_patient_timeline_html,
            inputs=[patient_dropdown],
            outputs=[timeline_plot, chart_status, chart_info, xai_list]
        ).then(
            _refresh_notes, inputs=[patient_dropdown], outputs=[note_selector, note_gradient]
        ).then(
            _refresh_readmission, inputs=[patient_dropdown],
            outputs=[readmission_curve, readmission_table]
        )

        # Draw the gradient bar for the selected note.
        note_selector.change(
            app.get_note_gradient_fig,
            inputs=[patient_dropdown, note_selector],
            outputs=[note_gradient]
        )

        # ====================================================================
        # EVENT HANDLERS - LABELLING MODE TAB
        # ====================================================================
        
        # Create hidden elements needed for event handlers (required by Gradio)
        current_view_info = gr.Textbox(label="Current View", value="", interactive=False, visible=False)
        view_back_btn = gr.Button("◀ Previous Month", variant="secondary", visible=False)
        view_forward_btn = gr.Button("Next Month ▶", variant="secondary", visible=False)
        
        # Load patient for labelling: Sets up monthly labelling interface
        load_labelling_btn.click(
            app.load_patient_for_labelling,  # App method that prepares patient for monthly labelling
            inputs=[patient_dropdown_label],
            outputs=[label_chart_status, month_dropdown, current_view_info_visible, labels_info]
        )
        
        # --- MONTH NAVIGATION FUNCTIONS ---
        # These functions handle navigation between months in the labelling interface
        
        def navigate_month_back(month):
            """Navigate to previous month in labelling dropdown"""
            new_month = app.navigate_month(month, "back")
            edit_choices = app.get_monthly_labels_list()  # Update edit dropdown
            # Reset form when changing labeling month
            return new_month, gr.update(choices=edit_choices), "No", [], ""
        
        def navigate_month_forward(month):
            """Navigate to next month in labelling dropdown"""
            new_month = app.navigate_month(month, "forward")
            edit_choices = app.get_monthly_labels_list()  # Update edit dropdown
            # Reset form when changing labeling month
            return new_month, gr.update(choices=edit_choices), "No", [], ""        
        def update_month_view(month):
            """Update monthly timeline view when month selection changes"""
            app.current_view_offset = 0  # Reset view offset to show selected month
            if month:
                fig, status, view_info = app.update_monthly_view(month, 0)
                edit_choices = app.get_monthly_labels_list()
                return fig, status, view_info, gr.update(choices=edit_choices), "No", [], ""
            return None, "No month selected", "", gr.update(choices=[]), "No", [], ""
        
        # Month navigation button event handlers
        month_back_btn.click(
            navigate_month_back,
            inputs=[month_dropdown],
            outputs=[month_dropdown, edit_month_dropdown, flare_evidence, category_dropdown_label, reason_input_label]
        )
        
        month_forward_btn.click(
            navigate_month_forward,
            inputs=[month_dropdown],
            outputs=[month_dropdown, edit_month_dropdown, flare_evidence, category_dropdown_label, reason_input_label]

        )
        
        # Month dropdown change: Updates timeline view and edit options
        month_dropdown.change(
            update_month_view,
            inputs=[month_dropdown],
            outputs=[monthly_timeline_plot, label_chart_status, current_view_info, edit_month_dropdown, 
             flare_evidence, category_dropdown_label, reason_input_label]
        )
        
        # --- TIMELINE VIEW NAVIGATION ---
        # These buttons allow viewing different months while keeping labelling target separate
        
        # View navigation for the visible buttons (above timeline)
        view_back_btn_visible.click(
            lambda month: app.navigate_view(month, "back"),  # Navigate view backward
            inputs=[month_dropdown],
            outputs=[monthly_timeline_plot, label_chart_status, current_view_info_visible]
        )
        
        view_forward_btn_visible.click(
            lambda month: app.navigate_view(month, "forward"),  # Navigate view forward
            inputs=[month_dropdown],
            outputs=[monthly_timeline_plot, label_chart_status, current_view_info_visible]
        )
        
        # Hidden button event handlers (needed for internal state management)
        view_back_btn.click(
            lambda month: app.navigate_view(month, "back"),
            inputs=[month_dropdown],
            outputs=[monthly_timeline_plot, label_chart_status, current_view_info]
        )
        
        view_forward_btn.click(
            lambda month: app.navigate_view(month, "forward"),
            inputs=[month_dropdown],
            outputs=[monthly_timeline_plot, label_chart_status, current_view_info]
        )
        
        # --- LABEL MANAGEMENT FUNCTIONS ---
        # These functions handle saving, clearing, and editing flare labels
        
        def save_label_and_refresh(month, evidence, categories, reason):
            status, labels_info = app.save_monthly_label(month, evidence, categories, reason)
            # Refresh the timeline to show new flare highlighting
            if month:
                # Use the existing update_monthly_view method instead
                fig, chart_status, view_info = app.update_monthly_view(month, app.current_view_offset)
                edit_choices = app.get_monthly_labels_list()
                return status, labels_info, fig, gr.update(choices=edit_choices)
            return status, labels_info, None, gr.update(choices=[])
        
        def clear_label_and_refresh(month):
            """Clear monthly flare label and refresh timeline visualization"""
            status, labels_info = app.clear_monthly_label(month)
            # Refresh the timeline to remove flare highlighting
            if month:
                fig, chart_status, view_info = app.update_monthly_view(month, app.current_view_offset)
                edit_choices = app.get_monthly_labels_list()
                return status, labels_info, fig, gr.update(choices=edit_choices)
            return status, labels_info, None, gr.update(choices=[])
        
        # Save label button: Saves current month's flare label
        save_label_btn.click(
            save_label_and_refresh,
            inputs=[month_dropdown, flare_evidence, category_dropdown_label, reason_input_label],
            outputs=[label_chart_status, labels_info, monthly_timeline_plot, edit_month_dropdown]
        )
        
        # Clear label button: Removes current month's flare label
        clear_label_btn.click(
            clear_label_and_refresh,
            inputs=[month_dropdown],
            outputs=[label_chart_status, labels_info, monthly_timeline_plot, edit_month_dropdown]
        )
        
        # --- EDIT EXISTING LABELS ---
        
        # Load existing label for editing: Populates form with existing label data
        load_edit_btn.click(
            app.load_label_for_edit,  # App method that loads existing label data
            inputs=[edit_month_dropdown],  # Month to edit
            outputs=[flare_evidence, category_dropdown_label, reason_input_label, label_chart_status]
        )
        
        # Delete label button: Removes existing label completely
        delete_label_btn.click(
            app.delete_monthly_label,  # App method that deletes the label
            inputs=[edit_month_dropdown],
            outputs=[label_chart_status, labels_info]
        ).then(
            # Chain: After deletion, refresh the timeline view
            lambda month: app.update_monthly_view(month, app.current_view_offset) if month else (None, "No month selected", ""),
            inputs=[month_dropdown],
            outputs=[monthly_timeline_plot, label_chart_status, current_view_info_visible]
        ).then(
            # Chain: Update the edit dropdown choices (remove deleted month)
            lambda: gr.update(choices=app.get_monthly_labels_list()),
            outputs=[edit_month_dropdown]
        )
    
    return demo

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Create app instance with command line arguments
    app = create_app_with_args()
    
    # Add monthly labelling methods to the app class
    # (This extends the PatientTimelineApp class with additional methods)
    add_monthly_labelling_methods(PatientTimelineApp)
    
    # Create and launch the Gradio web interface
    demo = create_interface()
    demo.launch()  # By default launches on localhost:7860