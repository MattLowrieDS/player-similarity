import column_mappings as cm
import polars as pl
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics.pairwise import cosine_similarity

# Change the text color of the multiselect chips to black for better visibility
st.markdown(
    """
    <style>
    /* Target the text color inside the selection chips */
    .stMultiSelect span[data-baseweb="tag"] {
        color: #000000 !important;  /* Change this to your preferred text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def draw_radar(metric_names, percentiles, raw_values, player_name, descriptions):
  HOLE = 12          # empty centre (in percentile units)
  GAP = 0.03         # gap between bars (radians)

  n = len(metric_names)
  width = 2 * np.pi / n
  theta = np.arange(n) * width          # bar centres, starting at the top, going clockwise

  fig = plt.figure(figsize=(9, 10), facecolor='black')
  # ax = fig.add_axes([0.0, 0.08, 0.84, 0.80], projection='polar', facecolor='black')
  ax = fig.add_axes([0.04, 0.08, 0.6, 0.8], projection='polar', facecolor='black')
  ax.set_theta_zero_location('N')
  ax.set_theta_direction(-1)
  ax.set_ylim(0, 100 + HOLE)
  ax.axis('off')

  # Radar bars
  ax.bar(theta, percentiles, width=width - GAP, bottom=HOLE,
         color=MAIN_COLOR, edgecolor='black', linewidth=2, zorder=3, alpha=0.6)

  # Dashed percentile rings
  ring_t = np.linspace(0, 2 * np.pi, 400)
  for p in (25, 50, 75, 100):
      ax.plot(ring_t, np.full_like(ring_t, p + HOLE), ls='--', lw=1, color='grey', zorder=2)
      ax.text(np.deg2rad(360/n/2), p + HOLE, f'{p}%',
              ha='center', va='center', fontsize=7, fontweight='bold', color='white',
              bbox=dict(fc='black', ec='none', pad=1), zorder=4, alpha=0.8)

  # Outer labels
  def rot(angle_rad):
      """Tangential text rotation, flipped so it never reads upside down."""
      deg = -np.rad2deg(angle_rad) % 360      # screen angle of the tangent (clockwise axis)
      return deg + 180 if 90 < deg < 270 else deg

  for t, val, desc in zip(theta, raw_values, descriptions):
      r = 100 + HOLE
      # Inner metric values
      ax.text(t, r + 8, f'Metric score: {val:.2f}', rotation=rot(t), rotation_mode='anchor',
              ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=5,
              bbox=dict(fc='black', ec='none', pad=1))
      # Outer metric name
      ax.text(t, r + 18, desc, rotation=rot(t), rotation_mode='anchor',
              ha='center', va='center', fontsize=12, fontweight='bold', color=MAIN_COLOR)

  # Title and footer
  fig.text(0.35, 0.855, f'{player_name}', ha='center', fontsize=26, fontweight='bold', color='white')
  fig.text(-0.01, 0.08, 'Season: 2024/25\nCompetitions: A League\n', fontsize=7, va='bottom', color='white')
  st.pyplot(fig)


# Main color for the UI and radar charts
MAIN_COLOR = '#0ec80a'
PLAYER_COLS = ['player_id', 'player_name', 'player_short_name']
DEFAULT_SELECTED = ['pass_count_shotwithin10s_p30tip', 'pass_avgxpass_attempted', 'pass_count_linebreak_completed_p30tip',
                    'offballrun_count_targeted_p30tip', 'passopportunity_count_dangerous_p30tip', 'passopportunity_count_p30tip']

# Load the fonts for matplotlib
font_entry_med = fm.FontEntry(
    fname='Source_Sans_3/static/SourceSans3-Medium.ttf',
    name='Source Sans',
    weight='normal',
    style='normal',
)
fm.fontManager.ttflist.append(font_entry_med)
font_entry_bold = fm.FontEntry(
    fname='Source_Sans_3/static/SourceSans3-Bold.ttf',
    name='Source Sans',
    weight='bold',
    style='normal',
)
fm.fontManager.ttflist.append(font_entry_bold)
plt.rcParams['font.family'] = 'Source Sans'

# Load the preprocessed aggregated data
data_df = pl.read_parquet('aggregate_data.parquet')

st.title('Player Similarity')

player_names = data_df['player_name'].drop_nulls().unique().sort().to_list()
# Default by showing Adrian Segečić since he had a standout performance in the 2024/25 season
default_player = 'Adrian Segecic'
default_index = player_names.index(default_player) if default_player in player_names else 0

# Player select box
selected_player = st.selectbox('Select a player', player_names, index=default_index)

metric_cols = [col for col in data_df.columns if col not in PLAYER_COLS and not col.endswith('_right') and not col.endswith('_pct')]
# Use the long descriptions for the multiselect, but keep track of the corresponding column names
desc_to_col = {cm.metric_descriptions_long[col]: col for col in metric_cols if col in cm.metric_descriptions_long}
# Use the short descriptions for the radar chart, but keep track of the corresponding column names
abbr_to_col = {cm.metric_descriptions_short[col]: col for col in metric_cols if col in cm.metric_descriptions_short}

# Metric multiselect drop-down
selected_descriptions = st.multiselect('Select metrics', list(desc_to_col.keys()), max_selections=8,
                                       default=[cm.metric_descriptions_long[col] for col in DEFAULT_SELECTED])
selected_columns = [desc_to_col[desc] for desc in selected_descriptions]
selected_abbrs = [abbr for abbr, col in abbr_to_col.items() if col in selected_columns]

if selected_player and len(selected_columns) >= 6:
    player_data = data_df.filter(pl.col('player_name') == selected_player)
    
    if len(player_data) > 0:
        metric_names = selected_columns
        raw_values = player_data.select(metric_names).to_numpy().flatten().tolist()
        percentiles = player_data.select([f"{m}_pct" for m in metric_names]).to_numpy().flatten().tolist()
        
        draw_radar(metric_names, percentiles, raw_values, selected_player, selected_abbrs)

        st.subheader('Most Similar Players')

        metric_matrix = data_df.select(metric_names).to_numpy()
        player_indices = data_df.select(pl.col('player_name') == selected_player).to_numpy().flatten()
        selected_idx = np.where(player_indices)[0][0]
        player_vector = metric_matrix[selected_idx].reshape(1, -1)

        similarities = cosine_similarity(player_vector, metric_matrix).flatten()
        similar_indices = np.argsort(similarities)[::-1]

        top_3_players = []
        for idx in similar_indices:
            idx = int(idx)
            if idx != selected_idx and not np.isnan(similarities[idx]):
                sim_name = data_df['player_name'][idx]
                top_3_players.append((sim_name, similarities[idx]))
            if len(top_3_players) == 3:
                break

        for sim_player_name, sim_score in top_3_players:
            st.write(f'**{sim_player_name}** (similarity: {sim_score:.3f})')
            sim_player_data = data_df.filter(pl.col('player_name') == sim_player_name)
            sim_raw_values = sim_player_data.select(metric_names).to_numpy().flatten().tolist()
            sim_percentiles = sim_player_data.select([f"{m}_pct" for m in metric_names]).to_numpy().flatten().tolist()
            draw_radar(metric_names, sim_percentiles, sim_raw_values, sim_player_name, selected_abbrs)
else:
    st.info('Choose a player and at least 6 metrics to see the radar chart for that player.')

st.markdown(
    '<p style="text-align: center; margin-top: 2rem; color: #888;">'
    'Data by <a href="https://github.com/SkillCorner/opendata/tree/master" target="_blank" style="color: #0ec80a; text-decoration: none;">SkillCorner</a>'
    '</p>',
    unsafe_allow_html=True
)
