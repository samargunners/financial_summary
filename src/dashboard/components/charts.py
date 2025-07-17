"""Chart components for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional
from config.settings import DEFAULT_COLOR_PALETTE, DEFAULT_CHART_THEME


class ChartBuilder:
    """Build various chart types for the financial dashboard."""
    
    def __init__(self):
        self.color_palette = DEFAULT_COLOR_PALETTE
        self.theme = DEFAULT_CHART_THEME
    
    def create_revenue_trend_chart(self, df: pd.DataFrame,
                                  date_col: str = 'date',
                                  amount_col: str = 'amount') -> go.Figure:
        """Create revenue trend line chart.
        
        Args:
            df: DataFrame with financial data
            date_col: Name of date column
            amount_col: Name of amount column
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return self._create_empty_chart("No data available")
        
        # Group by date and sum amounts
        daily_revenue = df.groupby(date_col)[amount_col].sum().reset_index()
        daily_revenue = daily_revenue.sort_values(date_col)
        
        fig = px.line(
            daily_revenue,
            x=date_col,
            y=amount_col,
            title="Revenue Trend Over Time",
            template=self.theme
        )
        
        fig.update_traces(
            line=dict(color=self.color_palette[0], width=3),
            hovertemplate="<b>Date:</b> %{x}<br>" +
                         "<b>Revenue:</b> $%{y:,.2f}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode='x unified'
        )
        
        return fig
    
    def create_category_pie_chart(self, df: pd.DataFrame,
                                 category_col: str = 'category',
                                 amount_col: str = 'amount') -> go.Figure:
        """Create pie chart for category distribution.
        
        Args:
            df: DataFrame with financial data
            category_col: Name of category column
            amount_col: Name of amount column
            
        Returns:
            Plotly figure object
        """
        if df.empty or category_col not in df.columns:
            return self._create_empty_chart("No category data available")
        
        # Group by category and sum amounts
        category_totals = df.groupby(category_col)[amount_col].sum().reset_index()
        category_totals = category_totals.sort_values(amount_col, ascending=False)
        
        fig = px.pie(
            category_totals,
            names=category_col,
            values=amount_col,
            title="Revenue Distribution by Category",
            template=self.theme,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>" +
                         "Amount: $%{value:,.2f}<br>" +
                         "Percentage: %{percent}<br>" +
                         "<extra></extra>"
        )
        
        return fig
    
    def create_bar_chart(self, df: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        title: str = "Bar Chart",
                        color_col: Optional[str] = None) -> go.Figure:
        """Create a bar chart.
        
        Args:
            df: DataFrame with data
            x_col: Name of x-axis column
            y_col: Name of y-axis column
            title: Chart title
            color_col: Optional column for color mapping
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return self._create_empty_chart("No data available")
        
        if color_col and color_col in df.columns:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                template=self.theme,
                color_discrete_sequence=self.color_palette
            )
        else:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=title,
                template=self.theme,
                color_discrete_sequence=self.color_palette
            )
        
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Value: %{y:,.2f}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_heatmap(self, df: pd.DataFrame,
                      x_col: str,
                      y_col: str,
                      z_col: str,
                      title: str = "Heatmap") -> go.Figure:
        """Create a heatmap.
        
        Args:
            df: DataFrame with data
            x_col: Name of x-axis column
            y_col: Name of y-axis column
            z_col: Name of value column
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return self._create_empty_chart("No data available")
        
        # Pivot data for heatmap
        pivot_data = df.pivot_table(
            values=z_col,
            index=y_col,
            columns=x_col,
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate="<b>%{x}</b><br>" +
                         "<b>%{y}</b><br>" +
                         "Value: %{z:,.2f}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          title: str = "Scatter Plot",
                          color_col: Optional[str] = None,
                          size_col: Optional[str] = None) -> go.Figure:
        """Create a scatter plot.
        
        Args:
            df: DataFrame with data
            x_col: Name of x-axis column
            y_col: Name of y-axis column
            title: Chart title
            color_col: Optional column for color mapping
            size_col: Optional column for size mapping
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return self._create_empty_chart("No data available")
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            template=self.theme,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "%{y:,.2f}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_box_plot(self, df: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       title: str = "Box Plot") -> go.Figure:
        """Create a box plot.
        
        Args:
            df: DataFrame with data
            x_col: Name of x-axis column (categorical)
            y_col: Name of y-axis column (numerical)
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return self._create_empty_chart("No data available")
        
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template=self.theme,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    def create_multi_line_chart(self, df: pd.DataFrame,
                              x_col: str,
                              y_cols: List[str],
                              title: str = "Multi-Line Chart") -> go.Figure:
        """Create a multi-line chart.
        
        Args:
            df: DataFrame with data
            x_col: Name of x-axis column
            y_cols: List of y-axis column names
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return self._create_empty_chart("No data available")
        
        fig = go.Figure()
        
        for i, col in enumerate(y_cols):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode='lines',
                    name=col.replace('_', ' ').title(),
                    line=dict(color=self.color_palette[i % len(self.color_palette)], width=3),
                    hovertemplate="<b>%{x}</b><br>" +
                                 f"<b>{col.replace('_', ' ').title()}:</b> %{{y:,.2f}}<br>" +
                                 "<extra></extra>"
                ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        return fig
    
    def create_waterfall_chart(self, categories: List[str],
                              values: List[float],
                              title: str = "Waterfall Chart") -> go.Figure:
        """Create a waterfall chart.
        
        Args:
            categories: List of category names
            values: List of values
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if not categories or not values:
            return self._create_empty_chart("No data available")
        
        fig = go.Figure(go.Waterfall(
            name="Financial Flow",
            orientation="v",
            measure=["relative"] * (len(values) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_chart(self, message: str = "No data available") -> go.Figure:
        """Create an empty chart with a message.
        
        Args:
            message: Message to display
            
        Returns:
            Empty Plotly figure
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            template=self.theme,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
