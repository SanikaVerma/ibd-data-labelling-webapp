import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import textwrap
from datetime import datetime
import yaml
from rmt23345_events import load_all_events


def _load_study_config(config_path: str = "study_config.yaml") -> dict:
    """Load study_config.yaml if it exists; return empty dict otherwise."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[study_config] Warning: could not read {config_path}: {e}")
        return {}


class PatientTimelineApp:
    """
    Main application class for the Patient Timeline Viewer.
    
    This class manages patient medical event data, provides timeline visualizations,
    and allows doctors to mark flare periods in IBD patients. The app supports
    both date-based flare marking and monthly flare labeling.
    
    Key Features:
    - Load and filter patient medical event data
    - Create interactive timeline visualizations
    - Mark and save flare periods (date-based and monthly)
    - Support for different patient access groups
    - Export functionality for data and charts
    """
    
    def __init__(self, patient_range=None, group_name="Default", rmt23345_data_dir=None):
        """
        Initialize the Patient Timeline App.
        
        Args:
            patient_range (tuple, optional): (start_id, end_id) for patient filtering
            group_name (str): Name of the patient group for display purposes
        """
        # === Data Storage ===
        self.combined_data = None          # Main dataset containing all patient events
        self.current_patient_data = None   # Filtered data for currently selected patient
        self.current_patient_id = None     # ID of currently selected patient
        self.ranges = []                   # List of date-based flare periods: [(start_date, end_date, categories, reason), ...]
        self.fig = None                    # Current plotly figure object
        
        # === UI State Management ===
        self.editing_flare_index = None    # Index of flare currently being edited (for future use)
        
        # === Patient Access Control ===
        # These settings control which patients this app instance can access
        self.patient_range = patient_range  # (start_id, end_id) tuple or None for all patients
        self.group_name = group_name       # Display name for this patient group

        # === RMT23345 Data Directory ===
        # If set, load events from cleaned RMT23345 CSVs instead of the default CSV
        self.rmt23345_data_dir = rmt23345_data_dir
        
        # === Navigation State ===
        # Used in labeling mode to track view offset from selected month
        self.current_view_offset = 0       # Integer offset for month navigation
        
        # === Initialization ===
        # Automatically load data when app starts
        self.auto_load_data()
    
    def auto_load_data(self):
        """
        Load patient data from the RMT23345 directory if one was specified.
        If no directory is given, combined_data stays None — the user must
        either provide --rmt23345-dir at startup or upload files via the UI.
        """
        if self.rmt23345_data_dir and os.path.isdir(self.rmt23345_data_dir):
            try:
                study_config = _load_study_config()
                if study_config:
                    enabled = study_config.get("sources", {}).get("enabled", [])
                    print(f"study_config.yaml: loading sources {enabled}")
                print(f"Loading from RMT23345 directory: {self.rmt23345_data_dir}")
                df = load_all_events(self.rmt23345_data_dir, config=study_config)
                if len(df) > 0:
                    df['start_date'] = pd.to_datetime(df['start_date'], utc=True, errors='coerce').dt.tz_localize(None)
                    df['end_date']   = pd.to_datetime(df['end_date'],   utc=True, errors='coerce').dt.tz_localize(None)
                    self.combined_data = df
                    print(f"Loaded {len(df)} events from RMT23345 CSVs")
                    self.apply_patient_filter()
                else:
                    print("No events found in the specified directory.")
            except Exception as e:
                print(f"Failed to load RMT23345 data: {e}")
        else:
            print("No data directory specified. Use --rmt23345-dir")
    
    def apply_patient_filter(self):
        """no but my 
        Apply patient range filtering based on command line arguments.
        
        This method filters the loaded data to only include patients within
        the specified ID range. This is used for access control - different
        users/groups can be given access to different patient cohorts.
        
        The filtering is applied after data loading and logs the filtering results.
        """
        if self.combined_data is not None and self.patient_range is not None:
            start_id, end_id = self.patient_range
            
            # === Record Original Data Stats ===
            original_count = len(self.combined_data)
            original_patients = len(self.combined_data['patient_id'].unique())
            
            # === Apply Patient ID Filter ===
            # Filter to only include patients in the specified range
            self.combined_data = self.combined_data[
                (self.combined_data['patient_id'] >= start_id) & 
                (self.combined_data['patient_id'] <= end_id)
            ].copy()
            
            # === Record Filtered Data Stats ===
            filtered_count = len(self.combined_data)
            filtered_patients = len(self.combined_data['patient_id'].unique()) if filtered_count > 0 else 0
            
            # === Log Filtering Results ===
            print(f"Patient filtering applied ({self.group_name}):")
            print(f"  Range: Patient IDs {start_id} to {end_id}")
            print(f"  Records: {original_count} → {filtered_count}")
            print(f"  Patients: {original_patients} → {filtered_patients}")
    
    def get_data_status(self):
        """
        Get current data loading status for display in the UI.
        
        Returns:
            str: Formatted status string showing data loading state,
                 number of records, patients, and ID range if applicable
        """
        if self.combined_data is not None:
            patient_count = len(self.combined_data['patient_id'].unique())
            if self.patient_range:
                start_id, end_id = self.patient_range
                return f"Data loaded ({self.group_name}): {len(self.combined_data)} records for {patient_count} patients (IDs {start_id}-{end_id})"
            else:
                return f"Data loaded: {len(self.combined_data)} records for {patient_count} patients"
        else:
            return "No data loaded"
    
    def get_patient_choices(self):
        """
        Get list of available patient IDs for dropdown selection.
        
        Returns:
            list: List of patient IDs as strings, sorted numerically
        """
        if self.combined_data is not None:
            patient_ids = sorted(self.combined_data['patient_id'].unique())
            return [str(pid) for pid in patient_ids]
        else:
            return []
    
    def reload_data(self):
        """
        Reload data from the directory and update UI components.
        
        This method is called when the user clicks the "Reload Data" button.
        It re-runs the data loading process and updates the UI with new data.
        
        Returns:
            tuple: (status_message, updated_dropdown, status_for_chart_info)
        """
        import gradio as gr
        self.auto_load_data()
        patient_choices = self.get_patient_choices()
        status = self.get_data_status()
        return status, gr.update(choices=patient_choices, value=patient_choices[0] if patient_choices else None), status
    
    def export_data(self, format_choice):
        """
        Export current data to CSV or Excel format.
        
        Args:
            format_choice (str): Either "CSV" or "Excel"
            
        Returns:
            str: Status message indicating export success or failure
        """
        if self.combined_data is None:
            return "No data to export"
        
        try:
            # === Generate Timestamped Filename ===
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format_choice == "CSV":
                filename = f"patient_data_{timestamp}.csv"
                self.combined_data.to_csv(filename, index=False)
            else:  # Excel
                filename = f"patient_data_{timestamp}.xlsx"
                self.combined_data.to_excel(filename, index=False)
            
            return f"Data exported to {filename}"
            
        except Exception as e:
            return f"Failed to export data: {str(e)}"
    
    def load_config_file(self, config_file):
        """
        Accept an uploaded study_config.yaml, read it, and load data from
        the file paths specified in its sources section. Paths may be absolute
        (pointing anywhere on the server) or relative (resolved against data_dir).

        Returns:
            tuple: (status_msg, data_status, patient_dropdown_update, chart_info)
        """
        import gradio as gr

        if config_file is None:
            return (
                "No config file selected.",
                self.get_data_status(),
                gr.update(),
                self.get_data_status(),
            )

        src = config_file if isinstance(config_file, str) else config_file.name

        try:
            with open(src, 'r') as f:
                uploaded_config = yaml.safe_load(f) or {}
        except Exception as e:
            return (
                f"Failed to read config file: {e}",
                self.get_data_status(),
                gr.update(),
                self.get_data_status(),
            )

        try:
            df = load_all_events(data_dir=self.rmt23345_data_dir, config=uploaded_config)
            if len(df) == 0:
                return (
                    "Config loaded but no events found. Check that the file paths in your YAML exist on this server.",
                    self.get_data_status(),
                    gr.update(),
                    self.get_data_status(),
                )
            df['start_date'] = pd.to_datetime(df['start_date'], utc=True, errors='coerce').dt.tz_localize(None)
            df['end_date']   = pd.to_datetime(df['end_date'],   utc=True, errors='coerce').dt.tz_localize(None)
            self.combined_data = df
            self.apply_patient_filter()
        except Exception as e:
            return (
                f"Error loading data from config: {e}",
                self.get_data_status(),
                gr.update(),
                self.get_data_status(),
            )

        patient_choices = self.get_patient_choices()
        data_status     = self.get_data_status()
        return (
            f"Loaded {len(self.combined_data):,} events for {len(patient_choices)} patients.",
            data_status,
            gr.update(choices=patient_choices, value=patient_choices[0] if patient_choices else None),
            data_status,
        )

    def load_patient_timeline(self, patient_id, ibd_filter="All Events"):
        """
        Load timeline for selected patient with IBD filtering.
        
        This is the main method for the Timeline Viewer tab. It loads data for
        a specific patient, applies IBD filtering, loads existing flares, and
        creates the timeline visualization.
        
        Args:
            patient_id (str): Patient ID to load
            ibd_filter (str): Filter type - "All Events", "IBD Related Only", or "Non-IBD Related Only"
            
        Returns:
            tuple: (plotly_figure, status_message, chart_info_text)
        """
        # === Validation ===
        if self.combined_data is None:
            return None, "Please load patient data first", ""
        
        if not patient_id:
            return None, "Please select a patient ID", ""
        
        try:
            # === Filter Data for Selected Patient ===
            patient_data = self.combined_data[
                self.combined_data['patient_id'] == int(patient_id)
            ].copy()
            
            # === Apply IBD Filtering ===
            # Filter events based on IBD-related flag
            if ibd_filter == "IBD Related Only":
                patient_data = patient_data[patient_data['ibd_related'] == True]
            elif ibd_filter == "Non-IBD Related Only":
                patient_data = patient_data[patient_data['ibd_related'] == False]
            # "All Events" requires no additional filtering
            
            # === Update Instance State ===
            self.current_patient_data = patient_data
            self.current_patient_id = int(patient_id)
            
            # === Load Existing Flare Data ===
            # Load any previously saved flare periods for this patient
            self.load_existing_flares()
            
            # === Create Timeline Visualization ===
            fig = self.create_timeline()
            info = self.get_chart_info()
            
            status_msg = f"Chart loaded for Patient {self.current_patient_id} ({ibd_filter})"
            
            return fig, status_msg, info
            
        except Exception as e:
            return None, f"Failed to load patient timeline: {str(e)}", ""
    
    def load_existing_flares(self):
        """
        Load existing flare periods from JSON file.
        
        Each patient's flare data is saved in a separate JSON file named
        'patient_{patient_id}_flares.json'. This method loads and parses
        that data, handling both old and new format flare records.
        
        Flare Format:
        - Old: {start_date, end_date, reason}
        - New: {start_date, end_date, categories, reason}
        """
        save_file = f'patient_{self.current_patient_id}_flares.json'
        self.ranges = []  # Reset flare list
        
        if os.path.exists(save_file):
            try:
                with open(save_file, 'r') as f:
                    saved_ranges = json.load(f)
                    
                    # === Parse Each Flare Record ===
                    for flare in saved_ranges:
                        flare_start = pd.to_datetime(flare['start_date'])
                        flare_end = pd.to_datetime(flare['end_date'])
                        
                        # === Handle Format Compatibility ===
                        # Support both old format (without categories) and new format (with categories)
                        if 'categories' in flare:
                            # New format with categories
                            categories = flare['categories']
                            reason = flare.get('reason', '')
                            self.ranges.append((flare_start, flare_end, categories, reason))
                        else:
                            # Old format compatibility - add empty categories
                            reason = flare.get('reason', '')
                            self.ranges.append((flare_start, flare_end, [], reason))
                            
            except Exception as e:
                print(f"Error loading flares: {e}")
    
    def load_patient_for_labelling(self, patient_id):
        """
        Load patient for labelling mode.
        
        This method prepares a patient's data for the monthly labelling interface.
        Unlike the timeline viewer, this loads ALL events (no IBD filtering) and
        prepares the month navigation system.
        
        Args:
            patient_id (str): Patient ID to load for labelling
            
        Returns:
            tuple: (status_message, month_dropdown_update, view_info, labels_info)
        """
        import gradio as gr
        if self.combined_data is None:
            return "Please load patient data first", gr.update(choices=[]), "", ""
        
        if not patient_id:
            return "Please select a patient ID", gr.update(choices=[]), "", ""
        
        try:
            # === Filter Data for Selected Patient ===
            # Load ALL events for labelling (no IBD filtering in labelling mode)
            patient_data = self.combined_data[
                self.combined_data['patient_id'] == int(patient_id)
            ].copy()
            
            # === Update Instance State ===
            self.current_patient_data = patient_data
            self.current_patient_id = int(patient_id)
            
            # === Reset Navigation State ===
            # Reset view offset when loading new patient
            self.current_view_offset = 0
            
            # === Get Available Months ===
            month_choices = self.get_patient_months()
            
            status_msg = f"Patient {self.current_patient_id} loaded for labelling"
            
            return status_msg, gr.update(choices=month_choices, value=month_choices[0] if month_choices else None), "", ""
            
        except Exception as e:
            return f"Failed to load patient: {str(e)}", gr.update(choices=[]), "", ""
    
    def get_patient_months(self):
        """
        Get list of available months for selected patient - ONLY months with events.
        
        This method identifies all months that contain at least one medical event
        for the current patient. It's used in the labelling mode to populate the
        month selection dropdown with only relevant months.
        
        For events that span multiple months, all months in the span are included.
        
        Returns:
            list: List of month strings in "Month YYYY" format (e.g., "January 2020")
        """
        if self.current_patient_data is None or len(self.current_patient_data) == 0:
            return []
        
        # === Collect Months with Events ===
        months_with_events = set()
        
        for _, row in self.current_patient_data.iterrows():
            # === Add Start Date Month ===
            start_month = row['start_date'].to_period('M')
            months_with_events.add(start_month)
            
            # === Add End Date Month ===
            # (if different from start month)
            end_month = row['end_date'].to_period('M')
            months_with_events.add(end_month)
            
            # === Add Months for Multi-Month Events ===
            # For events spanning multiple months, add all months in between
            if start_month != end_month:
                current_month = start_month
                while current_month <= end_month:
                    months_with_events.add(current_month)
                    current_month += 1
        
        # === Sort and Format Months ===
        months_sorted = sorted(months_with_events)
        
        # Convert to readable format: "January 2020", "February 2020", etc.
        month_strings = [f"{month.strftime('%B %Y')}" for month in months_sorted]
        return month_strings
    
    def get_all_patient_months(self):
        """
        Get list of ALL available months for selected patient - including empty months.
        
        This method generates all months between the first and last events,
        including months with no events. This is useful for comprehensive
        labelling where empty months might also need to be labeled.
        
        Returns:
            list: List of month strings including months without events
        """
        if self.current_patient_data is None or len(self.current_patient_data) == 0:
            return []
        
        # === Get Full Date Range ===
        min_date = self.current_patient_data['start_date'].min()
        max_date = self.current_patient_data['end_date'].max()
        
        # === Generate All Months in Range ===
        start_month = min_date.to_period('M')
        end_month = max_date.to_period('M')
        
        all_months = []
        current_month = start_month
        while current_month <= end_month:
            all_months.append(current_month)
            current_month += 1
        
        # === Format as Readable Strings ===
        month_strings = [f"{month.strftime('%B %Y')}" for month in all_months]
        return month_strings
    
    def navigate_month(self, current_month, direction):
        """
        Navigate to previous or next month - ONLY months with events.
        
        This method handles month navigation in the labelling interface,
        moving between months that actually contain medical events.
        
        Args:
            current_month (str): Currently selected month in "Month YYYY" format
            direction (str): "back" or "forward"
            
        Returns:
            str: New month selection or current month if navigation not possible
        """
        if not current_month:
            return current_month
        
        try:
            # === Get Available Months ===
            # Only get months with events for navigation
            month_choices = self.get_patient_months()
            
            # === Handle Month Not in List ===
            if current_month not in month_choices:
                # If current month is not in the filtered list, return the first available month
                return month_choices[0] if month_choices else None
            
            # === Navigate Between Months ===
            current_index = month_choices.index(current_month)
            
            if direction == "back" and current_index > 0:
                return month_choices[current_index - 1]
            elif direction == "forward" and current_index < len(month_choices) - 1:
                return month_choices[current_index + 1]
            else:
                # Can't navigate further in that direction
                return current_month
                
        except Exception as e:
            print(f"Error navigating month: {e}")
            return current_month
    
    def update_monthly_view(self, selected_month, view_offset=0):
        """
        Update the monthly timeline view - can show empty months.
        
        This method creates the monthly timeline visualization. The selected_month
        is what the user is labelling, while the view can be offset to show
        adjacent months for context.
        
        Args:
            selected_month (str): Month being labeled ("Month YYYY" format)
            view_offset (int): Offset from selected month for viewing (-1, 0, +1, etc.)
            
        Returns:
            tuple: (plotly_figure, status_message, view_month_string)
        """
        if not selected_month:
            return None, "No month selected", ""
        
        try:
            # === Import Timeline Creation Function ===
            # This function is defined in a separate module for better organization
            from timeline_visualization import create_monthly_timeline
            fig = create_monthly_timeline(self, selected_month, view_offset)
            
            # === Calculate Current View Month ===
            selected_period = pd.to_datetime(selected_month, format='%B %Y').to_period('M')
            view_period = selected_period + view_offset
            view_month_str = view_period.strftime('%B %Y')
            
            if fig:
                # === Check if View Month Has Events ===
                event_months = self.get_patient_months()
                
                # === Create Status Message ===
                if view_month_str in event_months:
                    status_msg = f"Showing timeline for {view_month_str} (Labelling: {selected_month})"
                else:
                    # Indicate when viewing a month with no events
                    status_msg = f"Showing timeline for {view_month_str} (NO EVENTS - Labelling: {selected_month})"
                
                return fig, status_msg, view_month_str
            else:
                return None, f"No data available for {view_month_str}", view_month_str
                
        except Exception as e:
            return None, f"Failed to update view: {str(e)}", ""
    
    def navigate_view(self, selected_month, direction):
        """
        Navigate the view forward or backward - can view ALL months including empty ones.
        
        This method changes which month is displayed in the timeline while keeping
        the same month selected for labelling. Users can view context around the
        month they're labelling.
        
        Args:
            selected_month (str): Month currently selected for labelling
            direction (str): "back" or "forward"
            
        Returns:
            tuple: (plotly_figure, status_message, view_month_string)
        """
        # === Update View Offset ===
        if direction == "back":
            self.current_view_offset -= 1
        elif direction == "forward":
            self.current_view_offset += 1
        
        # === Update View with New Offset ===
        return self.update_monthly_view(selected_month, self.current_view_offset)
    
    def create_timeline(self):
        """
        Create the plotly timeline figure.
        
        This method delegates to an external function for creating the main
        timeline visualization. The actual implementation is in timeline_visualization.py
        for better code organization.
        
        Returns:
            plotly.graph_objects.Figure: The timeline chart
        """
        from timeline_visualization import create_main_timeline
        return create_main_timeline(self)
    
    def get_chart_info(self):
        """
        Get chart information text for display in the UI.
        
        This method generates a comprehensive summary of the current patient's
        data, including event counts, date ranges, and flare information.
        It combines information from both date-based flares and monthly labels.
        
        Returns:
            str: Formatted information text for display
        """
        if self.current_patient_data is None:
            return "No patient data loaded"
        
        # === Import Label Mapping ===
        # Get the mapping from internal event names to readable display names
        from timeline_visualization import get_label_mapping
        label_mapping = get_label_mapping()
        
        # === Basic Patient Information ===
        info = f"Patient ID: {self.current_patient_id}\n"
        info += f"Total Events: {len(self.current_patient_data)}\n"
        info += f"Date Range: {self.current_patient_data['start_date'].min().date()} to {self.current_patient_data['end_date'].max().date()}\n\n"
        
        # === Event Type Summary ===
        info += "Event Types:\n"
        event_counts = self.current_patient_data['event_type'].value_counts()
        for event_type, count in event_counts.items():
            # Convert internal event names to readable format
            readable_name = label_mapping.get(event_type, event_type.replace('_', ' ').title())
            info += f"  {readable_name}: {count}\n"
        
        # === Date-Based Flare Periods ===
        # These are flares marked using the date range selection method
        info += f"\nDate-based Flares: {len(self.ranges)}\n"
        for i, flare_data in enumerate(self.ranges):
            # === Handle Both Old and New Flare Formats ===
            if len(flare_data) == 3:  # Old format: (start, end, reason)
                start_date, end_date, reason = flare_data
                categories = []
            else:  # New format: (start, end, categories, reason)
                start_date, end_date, categories, reason = flare_data
            
            # === Format Flare Information ===
            info += f"  {i+1}. {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}"
            
            # Add categories if available
            if categories:
                readable_categories = [label_mapping.get(cat, cat.replace('_', ' ').title()) for cat in categories]
                info += f" ({', '.join(readable_categories)})"
            
            # Add reason if available
            if reason:
                info += f": {reason}"
            
            info += "\n"
        
        # === Monthly Flares from Labelling Mode ===
        # These are flares marked using the monthly labelling interface
        monthly_labels = self.load_monthly_labels()
        # Filter to only show months marked as having evidence of flares
        monthly_flares = {k: v for k, v in monthly_labels.items() if v.get('evidence') == 'Yes'}
        
        info += f"\nMonthly Flares: {len(monthly_flares)}\n"
        for month_period_str, label_data in sorted(monthly_flares.items()):
            try:
                # === Format Monthly Flare Information ===
                period = pd.Period(month_period_str)
                month_str = period.strftime('%B %Y')
                info += f"  • {month_str}"
                
                # === Add Categories if Available ===
                categories = label_data.get('categories', [])
                if categories:
                    readable_categories = [label_mapping.get(cat, cat.replace('_', ' ').title()) for cat in categories]
                    info += f" ({', '.join(readable_categories)})"
                
                # === Add Reason if Available ===
                reason = label_data.get('reason', '')
                if reason:
                    info += f": {reason}"
                
                info += "\n"
                
            except Exception as e:
                print(f"Error formatting monthly flare: {e}")
                continue

        return info

    def get_xai_list_html(self, patient_id, n=15):
        """
        Render the top-n importance-scored items for a patient as a simple HTML list.

        Reads the TransEHR2 XAI score arrays on disk (via xai_arrays.XaiArrayReader)
        and maps each score back to its feature, decoded value and time — the real
        numpy-array workflow, replacing the earlier CSV mock-up.

        Importance scores are still placeholder values (no trained model yet), and
        the arrays are synthetic (real ones aren't available). Time is shown as
        hours since admission because recovering calendar dates needs each
        episode's admission timestamp (open question to the model author).
        No hover-to-highlight yet — planned for the later JavaScript overhaul.
        """
        if not patient_id:
            return "<p style='color:#6b7280;padding:12px;'>Select a patient to see importance scores.</p>"

        from xai_arrays import XaiArrayReader
        study_config = _load_study_config()
        xai_cfg = study_config.get("xai_arrays") or {}
        base_dir = xai_cfg.get("dir")
        suffix = xai_cfg.get("suffix", "train")
        if not base_dir:
            return "<p style='color:#6b7280;padding:12px;'>No xai_arrays directory configured in study_config.yaml.</p>"

        try:
            reader = XaiArrayReader(base_dir, suffix)
        except Exception as e:
            return f"<p style='color:#ef4444;padding:12px;'>Could not load XAI arrays: {e}</p>"

        # For the current synthetic data, episode_id == patient_id. Whether the
        # real episode IDs match the webapp's patient IDs is an open question.
        try:
            episode_id = int(patient_id)
        except (ValueError, TypeError):
            episode_id = patient_id

        if episode_id not in reader.episode_ids:
            return "<p style='color:#6b7280;padding:12px;'>No importance-score data for this patient.</p>"

        items = reader.top_records(episode_id, n=n)
        if not items:
            return "<p style='color:#6b7280;padding:12px;'>No importance-score data for this patient.</p>"

        rows = ""
        for it in items:
            hrs = it["time_hours"]
            time_str = f"{hrs:.0f} h" if hrs is not None else ""
            rows += (
                "<tr>"
                f"<td style='padding:6px 10px;'>{it['score']:+.3f}</td>"
                f"<td style='padding:6px 10px;'>{time_str}</td>"
                f"<td style='padding:6px 10px;'>{it['kind']}</td>"
                f"<td style='padding:6px 10px;'>{it['feature']}</td>"
                f"<td style='padding:6px 10px;'>{it['value']}</td>"
                "</tr>"
            )

        th = ("padding:6px 10px;color:#111827;background:#f3f4f6;"
              "border-bottom:2px solid #d1d5db;font-weight:600;")
        return f"""
        <div style='max-height:340px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:8px;'>
        <table style='width:100%;border-collapse:collapse;font-size:13px;'>
            <thead><tr style='text-align:left;position:sticky;top:0;'>
                <th style='{th}'>Score</th>
                <th style='{th}'>Time</th>
                <th style='{th}'>Kind</th>
                <th style='{th}'>Feature</th>
                <th style='{th}'>Value</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        <p style='color:#9ca3af;font-size:11px;padding:6px 2px 0;'>
        Synthetic test data — NOT derived from this real patient; shown only to
        develop the layout until the real model output exists. Scores are
        placeholders; time is hours since admission.
        </p>
        """

    def _xai_reader_and_episode(self, patient_id):
        """Build an XaiArrayReader from study_config and resolve the episode id.

        Returns (reader, episode_id), or (None, None) if unavailable.
        """
        if not patient_id:
            return None, None
        from xai_arrays import XaiArrayReader
        cfg = (_load_study_config().get("xai_arrays")) or {}
        base_dir = cfg.get("dir")
        if not base_dir:
            return None, None
        try:
            reader = XaiArrayReader(base_dir, cfg.get("suffix", "train"))
        except Exception:
            return None, None
        try:
            episode_id = int(patient_id)
        except (ValueError, TypeError):
            episode_id = patient_id
        if episode_id not in reader.episode_ids:
            return None, None
        return reader, episode_id

    def note_choices(self, patient_id):
        """Dropdown choices for the note gradient bar: (label, note_index) pairs."""
        reader, episode_id = self._xai_reader_and_episode(patient_id)
        if reader is None:
            return []
        try:
            notes = reader.text_records(episode_id)
        except Exception:
            return []
        choices = []
        for i, rec in enumerate(notes):
            t = rec.get("time_hours")
            when = f"{t:.0f} h" if t is not None else "?"
            choices.append((f"Note {i + 1} — {rec['feature']} @ {when}", i))
        return choices

    @staticmethod
    def _div_text_color(t):
        """Diverging text colour for a normalised score t in [-1, 1].

        Gray at the centre (unimportant, but still legible) saturating to blue
        for negative and red for positive — magnitude, not sign, means important.
        """
        t = max(-1.0, min(1.0, t))
        gray = (148, 163, 184)
        if t <= 0:
            end, f = (37, 99, 235), -t          # -> blue
        else:
            end, f = (220, 38, 38), t           # -> red
        r = int(gray[0] + f * (end[0] - gray[0]))
        g = int(gray[1] + f * (end[1] - gray[1]))
        b = int(gray[2] + f * (end[2] - gray[2]))
        return f"rgb({r},{g},{b})"

    def get_note_gradient_fig(self, patient_id, note_idx):
        """Diverging gradient bar for one note: each word sits under the portion
        of a blue-white-red scale that matches its importance score.

        Magnitude means importance: strongly blue (negative) and strongly red
        (positive) words are important; pale/centre words are not. Scores aren't
        printed on the words (position + colour carry the meaning) but appear on
        hover. The bar is always drawn — even for a patient with no note.
        """
        import numpy as np
        reader, episode_id = self._xai_reader_and_episode(patient_id)

        words, rec, has_any_notes = [], None, False
        if reader is not None:
            try:
                notes = reader.text_records(episode_id)
                has_any_notes = len(notes) > 0
                if note_idx is not None and note_idx != "":
                    rec = notes[int(note_idx)]
                    words = [w for w in rec["tokens"] if not w["is_special"]]
            except Exception:
                words, rec = [], None

        scores = np.array([w["score"] for w in words], dtype=float)
        vmax = float(np.max(np.abs(scores))) if words else 1.0
        vmax = vmax or 1.0
        lanes = 5

        fig = go.Figure()

        # The gradient bar: a horizontal blue -> white -> red strip.
        grid = np.linspace(-vmax, vmax, 201)
        fig.add_trace(go.Heatmap(
            x=grid, y=[0, 1], z=[grid.tolist(), grid.tolist()],
            zmin=-vmax, zmax=vmax, zmid=0,
            colorscale=[[0.0, "rgb(37,99,235)"], [0.5, "rgb(245,245,245)"],
                        [1.0, "rgb(220,38,38)"]],
            showscale=False, hoverinfo="skip",
        ))

        # Words placed under their score, staggered into lanes to avoid overlap.
        # Hover shows the score, not the word.
        if words:
            order = list(np.argsort(scores))           # left (neg) -> right (pos)
            lane_of = {int(idx): rank % lanes for rank, idx in enumerate(order)}
            ys = [-(1.0 + lane_of[i]) for i in range(len(words))]
            colors = [self._div_text_color(s / vmax) for s in scores]
            fig.add_trace(go.Scatter(
                x=scores, y=ys, mode="text",
                text=[(w["token"].strip() or w["token"]) for w in words],
                textfont=dict(color=colors, size=13),
                hovertext=[f"{w['score']:+.3f}" for w in words],
                hoverinfo="text",
            ))
        else:
            msg = ("Select a note above" if has_any_notes
                   else "No clinical note for this patient")
            fig.add_annotation(x=0, y=-2.5, text=msg, showarrow=False,
                               font=dict(size=12, color="#9ca3af"))

        # Importance labels sit ABOVE the bar (just under the plot title), with a
        # clear gap so they don't touch the gradient strip.
        y_bottom = -(1.0 + (lanes - 1))
        for xpos, label in [(-vmax, "important"), (0, "not important"),
                            (vmax, "important")]:
            fig.add_annotation(x=xpos, y=1.8, text=label, showarrow=False,
                               font=dict(size=11, color="#6b7280"))

        title = (f"{rec['feature']} @ {rec['time_hours']:.0f} h — words by importance"
                 if rec else "Clinical note — words by importance")
        fig.update_layout(
            title=title, height=300, margin=dict(l=20, r=20, t=40, b=20),
            # Match the timeline viewer's font.
            font=dict(family="Inter, Arial, sans-serif"),
            xaxis=dict(range=[-vmax * 1.15, vmax * 1.15], showticklabels=False,
                       zeroline=False, showgrid=False),
            yaxis=dict(range=[y_bottom - 0.6, 2.1], showticklabels=False,
                       zeroline=False, showgrid=False),
            showlegend=False, plot_bgcolor="white",
            # Gradio's narrow plot panel wraps a full modebar onto two rows, so we
            # keep only the two most useful tools — they fit on one line, top-right.
            modebar=dict(orientation="h",
                         remove=["select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                 "autoScale2d", "zoom2d", "pan2d"]),
        )
        return fig

    def load_patient_timeline_html(self, patient_id):
        """
        Load timeline as client-side HTML/JS (Plotly.js) — no IBD filter round-trip.

        Returns:
            tuple: (html_string, status_message, chart_info_text, xai_list_html)
        """
        if self.combined_data is None:
            placeholder = "<p style='color:#6b7280;padding:20px;'>Please load patient data first.</p>"
            return placeholder, "Please load patient data first", "", ""

        if not patient_id:
            placeholder = "<p style='color:#6b7280;padding:20px;'>Please select a patient ID.</p>"
            return placeholder, "Please select a patient ID", "", ""

        try:
            try:
                pid = int(patient_id)
            except (ValueError, TypeError):
                pid = patient_id

            patient_data = self.combined_data[
                self.combined_data['patient_id'].astype(str) == str(pid)
            ].copy()

            self.current_patient_data = patient_data
            self.current_patient_id   = pid
            self.load_existing_flares()

            from timeline_visualization import build_timeline_html
            html     = build_timeline_html(self)
            info     = self.get_chart_info()
            xai_html = self.get_xai_list_html(pid)
            status   = f"Chart loaded for Patient {self.current_patient_id} — filter using the buttons above the chart"
            return html, status, info, xai_html

        except Exception as e:
            err_html = f"<p style='color:#ef4444;padding:20px;'>Error: {str(e)}</p>"
            return err_html, f"Failed to load patient timeline: {str(e)}", "", ""