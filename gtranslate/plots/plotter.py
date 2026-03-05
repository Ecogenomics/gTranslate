import os
import logging

from gtranslate.biolib_lite.common import make_sure_path_exists


class FeaturePlotter:
    """Generates an interactive HTML 2D/3D scatter plot from the classifier features."""

    def __init__(self, tsv_path: str, out_file: str):
        self.tsv_path = tsv_path
        self.html_path = out_file
        # make sure the self.html_path ends with .html
        if not self.html_path.endswith('.html'):
            self.html_path += '.html'
        self.logger = logging.getLogger('timestamp')


    def generate_html(self):
        """Reads the TSV and generates the standalone HTML interactive plot."""
        if not os.path.exists(self.tsv_path):
            self.logger.warning(f"Cannot generate plot: {self.tsv_path} not found.")
            return

        try:
            import pandas as pd
            import plotly.express as px
        except ImportError:
            self.logger.warning(
                "Optional dependencies 'pandas' and 'plotly' are required to generate the HTML dashboard. Skipping plot generation.")
            return

        try:
            # 1. Load the data
            df = pd.read_csv(self.tsv_path, sep='\t')

            if df.empty:
                self.logger.warning("Feature file is empty. Skipping plot generation.")
                return

            # Ensure the predicted table is treated as a discrete category
            color_col = None
            categories = []
            cat_orders = None

            if 'predicted_tln_table' in df.columns:
                df['predicted_tln_table'] = df['predicted_tln_table'].astype(str)
                color_col = 'predicted_tln_table'

                # lets always have the same colors, even if some categories are missing from the current dataset
                categories = ['11', '4', '25']

                cat_orders = {color_col: categories}

            # 2. Get the feature columns
            ignore_cols = {'user_genome', 'predicted_tln_table'}
            features = [col for col in df.columns if col not in ignore_cols]

            if len(features) < 2:
                self.logger.warning("Not enough features to generate a plot (need at least 2).")
                return

            def create_buttons(dimension, is_3d=True):
                buttons = []
                for f in features:
                    if color_col:
                        trace_data = [df[df[color_col] == cat][f].values for cat in categories]
                    else:
                        trace_data = [df[f].values]

                    layout_update = {}
                    if is_3d:
                        layout_update[f'scene.{dimension}axis.title.text'] = f
                    else:
                        layout_update[f'{dimension}axis.title.text'] = f

                    # 'update' method accepts [data_updates, layout_updates]
                    buttons.append(dict(
                        method='update',
                        label=f,
                        args=[{dimension: trace_data}, layout_update]
                    ))
                return buttons

            if len(features) >= 3:
                fig = px.scatter_3d(
                    df, x=features[0], y=features[1], z=features[2],
                    color=color_col, hover_name='user_genome',
                    category_orders=cat_orders, title="gTranslate Feature Explorer"
                )

                fig.update_layout(
                    margin=dict(l=250),
                    updatemenus=[
                        dict(buttons=create_buttons('x', True), direction="down", x=-0.25, y=0.9, xanchor="left",
                             yanchor="top", showactive=True),
                        dict(buttons=create_buttons('y', True), direction="down", x=-0.25, y=0.7, xanchor="left",
                             yanchor="top", showactive=True),
                        dict(buttons=create_buttons('z', True), direction="down", x=-0.25, y=0.5, xanchor="left",
                             yanchor="top", showactive=True)
                    ],
                    annotations=[
                        dict(text="<b>X-Axis</b>", x=-0.25, y=0.92, xref="paper", yref="paper", showarrow=False,
                             xanchor="left", yanchor="bottom"),
                        dict(text="<b>Y-Axis</b>", x=-0.25, y=0.72, xref="paper", yref="paper", showarrow=False,
                             xanchor="left", yanchor="bottom"),
                        dict(text="<b>Z-Axis</b>", x=-0.25, y=0.52, xref="paper", yref="paper", showarrow=False,
                             xanchor="left", yanchor="bottom")
                    ]
                )

            else:
                fig = px.scatter(
                    df, x=features[0], y=features[1],
                    color=color_col, hover_name='user_genome',
                    category_orders=cat_orders, title="gTranslate Feature Explorer"
                )

                fig.update_layout(
                    margin=dict(l=250),
                    updatemenus=[
                        dict(buttons=create_buttons('x', False), direction="down", x=-0.25, y=0.9, xanchor="left",
                             yanchor="top", showactive=True),
                        dict(buttons=create_buttons('y', False), direction="down", x=-0.25, y=0.7, xanchor="left",
                             yanchor="top", showactive=True)
                    ],
                    annotations=[
                        dict(text="<b>X-Axis</b>", x=-0.25, y=0.92, xref="paper", yref="paper", showarrow=False,
                             xanchor="left", yanchor="bottom"),
                        dict(text="<b>Y-Axis</b>", x=-0.25, y=0.72, xref="paper", yref="paper", showarrow=False,
                             xanchor="left", yanchor="bottom")
                    ]
                )

            make_sure_path_exists(os.path.dirname(self.html_path))
            fig.write_html(self.html_path, include_plotlyjs="cdn")
            self.logger.info(f"Interactive HTML dashboard generated: {self.html_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate feature explorer plot: {e}")