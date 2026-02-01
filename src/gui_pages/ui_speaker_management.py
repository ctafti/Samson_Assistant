import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
import pytz

# Import backend functions from the main src directory
from src.speaker_profile_manager import get_all_speaker_profiles, update_speaker_profile, delete_speaker_profile
from src.matter_manager import get_all_matters
from src.logger_setup import logger
from main_orchestrator import queue_command_from_gui
from src.config_loader import get_config

# The `st.error` call has been removed from this function. A cached function MUST
# only return serializable data (like a list), not a Streamlit UI element.
@st.cache_data(ttl=60) # Cache speaker profiles for 60 seconds
def load_speaker_profiles() -> List[Dict[str, Any]]:
    """Loads and caches all speaker profiles."""
    logger.info("GUI: Loading speaker profiles for management page.")
    try:
        profiles = get_all_speaker_profiles()
        # Sort profiles by name for consistent display
        return sorted(profiles, key=lambda p: p.get('name', '').lower())
    except Exception as e:
        logger.error(f"GUI: Failed to load speaker profiles: {e}", exc_info=True)
        # On error, just return an empty list. The UI will handle showing the error message.
        return []

# MODIFICATION: Added a helper function to convert HEX to RGBA for transparent fills
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a HEX color string to an RGBA string with a given alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

def render_page():
    """
    Renders the Speaker Management page.
    """
    # <<< MODIFICATION: Refresh button moved to sidebar >>>
    with st.sidebar:
        st.header("Page Actions")
        if st.button("🔄 Refresh Speaker List", use_container_width=True):
            # Clear the cache and rerun the page to fetch fresh data
            load_speaker_profiles.clear()
            st.rerun()

    st.title("Speaker Management 🗣️")
    st.markdown("---")

    # --- Fetch all matters for the multiselect widgets ---
    all_matters = get_all_matters()
    matter_name_to_id = {m['name']: m['matter_id'] for m in all_matters}
    matter_id_to_name = {m['matter_id']: m['name'] for m in all_matters}

    # --- Get Timezone from Config for Display ---
    config = get_config()
    display_timezone_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
    try:
        display_tz = pytz.timezone(display_timezone_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(f"GUI: Unknown timezone '{display_timezone_str}' in config. Falling back to UTC.")
        display_tz = pytz.utc
    # --- END ---

    # <<< MODIFICATION: Removed columns, info box is now full-width >>>
    st.info(
        "View, edit, or delete enrolled speaker profiles. "
        "Changes to names or roles are saved instantly. Deleting a speaker is permanent and will require a database rebuild.",
        icon="ℹ️"
    )

    profiles = load_speaker_profiles()

    if not profiles:
        st.warning("Could not load any speaker profiles. This could be because none are enrolled yet, or there was an error reading the data file. Check the application logs for details.", icon="⚠️")
        st.stop()
    
    st.subheader(f"Found {len(profiles)} Enrolled Speaker Profiles")


    def render_profile_card(profile: Dict[str, Any], color: str, matter_name_to_id: Dict, matter_id_to_name: Dict):
        """
        Helper function to render a single speaker profile card.
        This function is defined locally to have access to the page's scope (and display_tz).
        """
        faiss_id = profile.get('faiss_id')
        name = profile.get('name', 'N/A')
        
        with st.container(border=True):
            # Add a colored bar at the top of the container for visual flair.
            st.markdown(
                f'<div style="background-color:{color};height:8px;border-radius:5px 5px 0 0;margin:-1rem -1rem 1rem -1rem;"></div>', 
                unsafe_allow_html=True
            )

            st.subheader(name)
            role = profile.get('role', 'Not set')
            
            # --- Convert created_utc to configured timezone ---
            created_utc_str = profile.get('created_utc', 'N/A')
            try:
                created_dt_utc = datetime.fromisoformat(created_utc_str.replace('Z', '+00:00'))
                created_dt_local = created_dt_utc.astimezone(display_tz)
                # <<< CHANGED DATE FORMAT >>>
                created_display = f"{created_dt_local.strftime('%B')} {created_dt_local.day}, {created_dt_local.year} at {created_dt_local.strftime('%H:%M')}"
            except (ValueError, AttributeError):
                created_display = created_utc_str

            # --- Create a full-width filled box for profile details ---
            light_fill_color = hex_to_rgba(color, 0.15)
            st.markdown(f"""
            <div style="background-color: {light_fill_color}; padding: 1rem; border-radius: 5px; margin: 0 -1rem 1rem -1rem;">
                <strong>Role:</strong> <code>{role}</code><br>
                <strong>FAISS ID:</strong> <code>{faiss_id}</code><br>
                <strong>Profile Created:</strong> <code>{created_display}</code>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Profile Strength**")

            # --- Lifetime Data (All-Time) ---
            permanent_log = profile.get('dynamic_threshold_feedback', [])
            lifetime_segment_count = len(permanent_log)
            lifetime_duration_s = profile.get('lifetime_total_audio_s', 0.0)

            st.markdown("<u>Lifetime Data (All-Time)</u>", unsafe_allow_html=True)
            if lifetime_segment_count > 0:
                col_life1, col_life2 = st.columns(2)
                with col_life1:
                    st.metric(label="📊 Feedback Events Logged", value=f"{lifetime_segment_count}")
                with col_life2:
                    st.metric(label="⏱️ Total Audio Contributed", value=f"{lifetime_duration_s:.1f}s")
            else:
                st.info("No lifetime data recorded yet. Process audio with this speaker to build their profile.", icon="🕰️")


            # --- Pending Evolution Data ---
            pending_data = profile.get('segment_embeddings_for_evolution', {})
            pending_segment_count = sum(len(segments) for segments in pending_data.values())
            pending_duration_s = sum(s.get('duration_s', 0.0) for segments in pending_data.values() for s in segments)

            st.markdown("<u>Pending Evolution Data (Since Last Recalc)</u>", unsafe_allow_html=True)
            if pending_segment_count > 0:
                col_pend1, col_pend2 = st.columns(2)
                with col_pend1:
                    st.metric(label="📊 New Segments Collected", value=f"{pending_segment_count}")
                with col_pend2:
                    st.metric(label="⏱️ New Audio Collected", value=f"{pending_duration_s:.1f}s")
            else:
                st.info("No new audio segments pending for the next profile evolution.", icon="✅")

            # --- Last Evolved Timestamp ---
            last_evolved_timestamps = profile.get('profile_last_evolved_utc', {})
            if isinstance(last_evolved_timestamps, dict) and last_evolved_timestamps:
                latest_timestamp_str = max(last_evolved_timestamps.values())
                try:
                    # Format the timestamp for display
                    latest_dt_utc = datetime.fromisoformat(latest_timestamp_str.replace('Z', '+00:00'))
                    latest_dt_local = latest_dt_utc.astimezone(display_tz) # Assumes display_tz is defined earlier in the function
                    evolved_display = f"{latest_dt_local.strftime('%B %d, %Y at %H:%M')}"
                    st.caption(f"Profile Last Evolved: `{evolved_display}`")
                except (ValueError, AttributeError):
                    st.caption(f"Profile Last Evolved: `{latest_timestamp_str}`")


            with st.expander("Edit or Delete Profile"):
                st.markdown("**Edit Profile**")
                new_name = st.text_input("Speaker Name", value=name, key=f"name_{faiss_id}")
                role_options = ["", "Owner", "Client", "Colleague", "Assistant", "Team Member", "Other"]
                current_role = profile.get('role', '')
                # Handle case where a role might exist but is not in the standard list
                try:
                    current_role_index = role_options.index(current_role) if current_role in role_options else 0
                except ValueError:
                    current_role_index = 0 # Default to empty if role is invalid
                new_role = st.selectbox(
                    "Assigned Role", 
                    options=role_options, 
                    index=current_role_index, 
                    key=f"role_{faiss_id}",
                    help="Assigning a role helps the system better understand task assignments and intent."
                )

                st.markdown("**Matter Heuristics**")
                st.caption("Associate this speaker with matters they are likely or unlikely to discuss. This helps the system make better automatic topic assignments.")
                
                # Get current relationships and default values for widgets
                relationships = profile.get('speaker_relationships', {})
                
                likely_ids = relationships.get('likely_matters', [])
                default_likely_names = [matter_id_to_name[m_id] for m_id in likely_ids if m_id in matter_id_to_name]

                unlikely_ids = relationships.get('unlikely_matters', [])
                default_unlikely_names = [matter_id_to_name[m_id] for m_id in unlikely_ids if m_id in matter_id_to_name]

                selected_likely_names = st.multiselect(
                    "Likely Matters",
                    options=list(matter_name_to_id.keys()),
                    default=default_likely_names,
                    key=f"likely_matters_{faiss_id}"
                )

                selected_unlikely_names = st.multiselect(
                    "Unlikely Matters",
                    options=list(matter_name_to_id.keys()),
                    default=default_unlikely_names,
                    key=f"unlikely_matters_{faiss_id}"
                )
                
                if st.button("Save Changes", key=f"save_{faiss_id}"):
                    update_payload = {}
                    if new_name != name:
                        update_payload['name'] = new_name
                    if new_role != current_role:
                        update_payload['role'] = new_role
                    
                    # Process matter heuristics and check for changes
                    selected_likely_ids = [matter_name_to_id[name] for name in selected_likely_names]
                    selected_unlikely_ids = [matter_name_to_id[name] for name in selected_unlikely_names]

                    current_relationships = profile.get('speaker_relationships', {})
                    current_likely = current_relationships.get('likely_matters', [])
                    current_unlikely = current_relationships.get('unlikely_matters', [])
                    
                    # Use sets for order-independent comparison to see if an update is needed
                    if set(selected_likely_ids) != set(current_likely) or set(selected_unlikely_ids) != set(current_unlikely):
                        update_payload['speaker_relationships'] = {
                            "likely_matters": selected_likely_ids,
                            "unlikely_matters": selected_unlikely_ids
                        }

                    if update_payload:
                        if update_speaker_profile(faiss_id, **update_payload):
                            st.toast(f"Profile for '{name}' updated successfully!", icon="✅")
                            load_speaker_profiles.clear()
                            st.rerun()
                        else:
                            st.error("Failed to update profile. Check logs.")
                    else:
                        st.toast("No changes to save.", icon="🤷")
                
                st.markdown("---")
                st.markdown("**Delete Profile**")
                st.warning(f"**Warning:** Deleting '{name}' is permanent. "
                           f"This will remove the speaker from the database.", icon="⚠️")
                
                confirm_delete = st.checkbox(f"I understand and want to delete '{name}'.", key=f"confirm_{faiss_id}")

                if st.button("DELETE PERMANENTLY", key=f"delete_{faiss_id}", disabled=not confirm_delete, type="primary"):
                    logger.info(f"GUI: User initiated deletion for speaker '{name}' (ID: {faiss_id}).")
                    if delete_speaker_profile(faiss_id):
                        st.toast(f"Profile for '{name}' deleted. Sending rebuild command...", icon="🗑️")
                        rebuild_command = {
                            "type": "REBUILD_SPEAKER_DATABASE",
                            "payload": {"reason": f"Deletion of speaker {name} (ID: {faiss_id}) from GUI."}
                        }
                        if queue_command_from_gui(rebuild_command):
                            logger.info("GUI: Sent REBUILD_SPEAKER_DATABASE command to backend.")
                            load_speaker_profiles.clear()
                            st.rerun()
                        else:
                            st.error("Failed to send rebuild command to backend.")
                    else:
                        st.error(f"Failed to delete profile for '{name}'. Check logs.")

    col1, col2 = st.columns(2)
    
    # Using light, accessible pastel colors that work well in both light and dark themes.
    COLOR_PALETTE = ["#BDE0FE", "#FFC8DD", "#CDB4DB", "#FFF2B2", "#B9FBC0", "#FFDAB9", "#E0BBE4"]
    
    for index, profile in enumerate(profiles):
        card_color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
        if index % 2 == 0:
            with col1:
                render_profile_card(profile, card_color, matter_name_to_id, matter_id_to_name)
        else:
            with col2:
                render_profile_card(profile, card_color, matter_name_to_id, matter_id_to_name)


    # --- Administrative Control Interface (unchanged) ---
    st.markdown("---")
    
    if 'maintenance_expander_state' not in st.session_state:
        st.session_state.maintenance_expander_state = False
    
    if 'recalc_select_all' not in st.session_state:
        st.session_state.recalc_select_all = False
    
    for profile in profiles:
        faiss_id = profile.get('faiss_id')
        checkbox_key = f"recalc_{faiss_id}"
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = False

    with st.expander("Advanced Profile Maintenance", expanded=st.session_state.maintenance_expander_state):
        st.markdown("### Speaker Profile Evolution Recalculation")
        st.info(
            "This feature recalculates a speaker's voice profile using all collected audio segments for a specific context. "
            "The new profile is a **recency-weighted average**, giving more importance to recent voice data while retaining long-term characteristics. "
            "This improves accuracy and helps the profile adapt over time.",
            icon="🔄"
        )
        
        st.markdown("**1. Select speakers to recalculate:**")
        
        col_sel1, col_sel2, _ = st.columns([1, 1, 2])
        
        with col_sel1:
            if st.button("Select All", key="select_all_recalc"):
                st.session_state.maintenance_expander_state = True
                for p in profiles:
                    st.session_state[f"recalc_{p.get('faiss_id')}"] = True
                st.rerun()
        
        with col_sel2:
            if st.button("Deselect All", key="deselect_all_recalc"):
                st.session_state.maintenance_expander_state = True
                for p in profiles:
                    st.session_state[f"recalc_{p.get('faiss_id')}"] = False
                st.rerun()
        
        mid_point_recalc = (len(profiles) + 1) // 2
        col_left, col_right = st.columns(2)
        
        with col_left:
            for profile in profiles[:mid_point_recalc]:
                faiss_id = profile.get('faiss_id')
                name = profile.get('name', 'N/A')
                st.checkbox(f"{name} (ID: {faiss_id})", key=f"recalc_{faiss_id}")
        
        with col_right:
            for profile in profiles[mid_point_recalc:]:
                faiss_id = profile.get('faiss_id')
                name = profile.get('name', 'N/A')
                st.checkbox(f"{name} (ID: {faiss_id})", key=f"recalc_{faiss_id}")
        
        st.markdown("---")
        
        st.markdown("**2. Select Context & Execute:**")
        
        st.warning(
            "⚠️ **Important:** This operation will temporarily lock the speaker database during "
            "recalculation. Large profile collections may take several minutes to process. "
            "Audio processing will be paused during this operation.",
            icon="⚠️"
        )
        
        recalc_context = st.selectbox(
            "Recalculate for Context:", 
            options=["in_person", "voip"], 
            key="recalc_context_select",
            help="Select the context for which the profiles should be recalculated. All selected speakers will be updated for this single context."
        )
        
        if st.button("Recalculate Selected Speaker Profiles", type="primary", key="execute_recalc"):
            st.session_state.maintenance_expander_state = True
            ids_to_recalculate = [p.get('faiss_id') for p in profiles if st.session_state.get(f"recalc_{p.get('faiss_id')}", False)]
            
            if not ids_to_recalculate:
                st.warning("No speakers selected for recalculation.", icon="⚠️")
            else:
                recalc_command = {
                    'type': 'RECALCULATE_SPEAKER_PROFILES',
                    'payload': {'targets': [{'faiss_id': fid, 'context': st.session_state.recalc_context_select} for fid in ids_to_recalculate]}
                }
                
                try:
                    if queue_command_from_gui(recalc_command):
                        selected_names = [p.get('name') for p in profiles if p.get('faiss_id') in ids_to_recalculate]
                        st.success(
                            f"✅ Recalculation command queued for context '{st.session_state.recalc_context_select}'! "
                            f"Processing {len(ids_to_recalculate)} speaker(s): {', '.join(selected_names)}. ",
                            icon="🚀"
                        )
                        logger.info(f"GUI: Queued RECALCULATE_SPEAKER_PROFILES command for {len(ids_to_recalculate)} speakers.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to queue recalculation command.", icon="🚨")
                        logger.error("GUI: Failed to queue RECALCULATE_SPEAKER_PROFILES command.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    logger.error(f"GUI: Exception while queuing recalculation command: {e}", exc_info=True)
        
        st.markdown("---")
        st.markdown("**ℹ️ About Profile Evolution:**")
        st.markdown(
            "- **Data Source:** Uses all high-confidence voice segments collected for the selected context.\n"
            "- **Weighting Method:** Uses a **recency-weighted** average. More recent segments have a greater influence, allowing the profile to adapt while maintaining long-term stability.\n"
            "- **Context-Specific:** Recalculation is performed for only one selected context (`IN_PERSON` or `VOIP`) at a time.\n"
            "- **Data Cleanup:** Upon successful recalculation, the used historical data for that context is cleared to prevent re-use.\n"
            "- **Database Safety:** The operation is atomic. The main speaker database is only replaced upon successful completion of the entire rebuild."
        )