"""Filter components for the dashboard."""

from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class FilterBuilder:
    """Build filter components for the dashboard."""
    
    def create_date_range_picker(self, 
                                component_id: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                display_format: str = 'YYYY-MM-DD') -> dcc.DatePickerRange:
        """Create a date range picker component.
        
        Args:
            component_id: Unique ID for the component
            start_date: Default start date
            end_date: Default end date
            display_format: Date display format
            
        Returns:
            Dash DatePickerRange component
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        return dcc.DatePickerRange(
            id=component_id,
            start_date=start_date,
            end_date=end_date,
            display_format=display_format,
            style={'width': '100%'}
        )
    
    def create_dropdown_filter(self,
                              component_id: str,
                              options: List[Dict[str, str]],
                              placeholder: str = "Select...",
                              multi: bool = False,
                              value: Optional[Any] = None) -> dcc.Dropdown:
        """Create a dropdown filter component.
        
        Args:
            component_id: Unique ID for the component
            options: List of options in format [{'label': 'Label', 'value': 'value'}]
            placeholder: Placeholder text
            multi: Allow multiple selections
            value: Default value(s)
            
        Returns:
            Dash Dropdown component
        """
        return dcc.Dropdown(
            id=component_id,
            options=options,
            placeholder=placeholder,
            multi=multi,
            value=value,
            style={'width': '100%'}
        )
    
    def create_slider_filter(self,
                            component_id: str,
                            min_value: float,
                            max_value: float,
                            step: float = 1,
                            value: Optional[float] = None,
                            marks: Optional[Dict[float, str]] = None) -> dcc.Slider:
        """Create a slider filter component.
        
        Args:
            component_id: Unique ID for the component
            min_value: Minimum slider value
            max_value: Maximum slider value
            step: Step size
            value: Default value
            marks: Optional marks dictionary
            
        Returns:
            Dash Slider component
        """
        if value is None:
            value = (min_value + max_value) / 2
        
        return dcc.Slider(
            id=component_id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    
    def create_range_slider_filter(self,
                                  component_id: str,
                                  min_value: float,
                                  max_value: float,
                                  step: float = 1,
                                  value: Optional[List[float]] = None,
                                  marks: Optional[Dict[float, str]] = None) -> dcc.RangeSlider:
        """Create a range slider filter component.
        
        Args:
            component_id: Unique ID for the component
            min_value: Minimum slider value
            max_value: Maximum slider value
            step: Step size
            value: Default range [min, max]
            marks: Optional marks dictionary
            
        Returns:
            Dash RangeSlider component
        """
        if value is None:
            value = [min_value, max_value]
        
        return dcc.RangeSlider(
            id=component_id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    
    def create_radio_filter(self,
                           component_id: str,
                           options: List[Dict[str, str]],
                           value: Optional[str] = None,
                           inline: bool = True) -> dcc.RadioItems:
        """Create a radio button filter component.
        
        Args:
            component_id: Unique ID for the component
            options: List of options in format [{'label': 'Label', 'value': 'value'}]
            value: Default value
            inline: Display options inline
            
        Returns:
            Dash RadioItems component
        """
        return dcc.RadioItems(
            id=component_id,
            options=options,
            value=value,
            inline=inline
        )
    
    def create_checklist_filter(self,
                               component_id: str,
                               options: List[Dict[str, str]],
                               value: Optional[List[str]] = None,
                               inline: bool = True) -> dcc.Checklist:
        """Create a checklist filter component.
        
        Args:
            component_id: Unique ID for the component
            options: List of options in format [{'label': 'Label', 'value': 'value'}]
            value: Default selected values
            inline: Display options inline
            
        Returns:
            Dash Checklist component
        """
        return dcc.Checklist(
            id=component_id,
            options=options,
            value=value or [],
            inline=inline
        )
    
    def create_search_filter(self,
                            component_id: str,
                            placeholder: str = "Search...",
                            debounce: bool = True) -> dbc.Input:
        """Create a search input filter component.
        
        Args:
            component_id: Unique ID for the component
            placeholder: Placeholder text
            debounce: Enable debounce for input
            
        Returns:
            Dash Bootstrap Input component
        """
        return dbc.Input(
            id=component_id,
            type="text",
            placeholder=placeholder,
            debounce=debounce,
            style={'width': '100%'}
        )
    
    def create_number_input_filter(self,
                                  component_id: str,
                                  min_value: Optional[float] = None,
                                  max_value: Optional[float] = None,
                                  step: float = 1,
                                  value: Optional[float] = None,
                                  placeholder: str = "Enter number...") -> dbc.Input:
        """Create a number input filter component.
        
        Args:
            component_id: Unique ID for the component
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            step: Step size
            value: Default value
            placeholder: Placeholder text
            
        Returns:
            Dash Bootstrap Input component
        """
        return dbc.Input(
            id=component_id,
            type="number",
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            placeholder=placeholder,
            style={'width': '100%'}
        )
    
    def create_filter_card(self,
                          title: str,
                          filter_component: Any,
                          description: Optional[str] = None) -> dbc.Card:
        """Wrap a filter component in a card layout.
        
        Args:
            title: Filter title
            filter_component: The filter component to wrap
            description: Optional description text
            
        Returns:
            Dash Bootstrap Card component
        """
        card_body = [
            html.H6(title, className="card-title"),
            filter_component
        ]
        
        if description:
            card_body.insert(1, html.P(description, className="card-text small text-muted"))
        
        return dbc.Card([
            dbc.CardBody(card_body)
        ], className="mb-3")
    
    def create_filter_section(self,
                             filters: List[Dict[str, Any]],
                             title: str = "Filters") -> dbc.Card:
        """Create a section containing multiple filters.
        
        Args:
            filters: List of filter dictionaries with 'title', 'component', and optional 'description'
            title: Section title
            
        Returns:
            Dash Bootstrap Card component containing all filters
        """
        filter_components = []
        
        for filter_def in filters:
            filter_card = self.create_filter_card(
                title=filter_def['title'],
                filter_component=filter_def['component'],
                description=filter_def.get('description')
            )
            filter_components.append(filter_card)
        
        return dbc.Card([
            dbc.CardHeader(html.H5(title)),
            dbc.CardBody(filter_components)
        ])
    
    def get_category_options(self, df: pd.DataFrame, 
                           category_col: str = 'category') -> List[Dict[str, str]]:
        """Get options for category filter from DataFrame.
        
        Args:
            df: DataFrame containing category data
            category_col: Name of category column
            
        Returns:
            List of options for dropdown
        """
        if df.empty or category_col not in df.columns:
            return []
        
        categories = df[category_col].dropna().unique()
        return [{'label': cat, 'value': cat} for cat in sorted(categories)]
    
    def get_store_options(self, df: pd.DataFrame,
                         store_col: str = 'store_location') -> List[Dict[str, str]]:
        """Get options for store filter from DataFrame.
        
        Args:
            df: DataFrame containing store data
            store_col: Name of store column
            
        Returns:
            List of options for dropdown
        """
        if df.empty or store_col not in df.columns:
            return []
        
        stores = df[store_col].dropna().unique()
        return [{'label': store, 'value': store} for store in sorted(stores)]
    
    def get_amount_range(self, df: pd.DataFrame,
                        amount_col: str = 'amount') -> Dict[str, float]:
        """Get min and max values for amount filter.
        
        Args:
            df: DataFrame containing amount data
            amount_col: Name of amount column
            
        Returns:
            Dictionary with min and max values
        """
        if df.empty or amount_col not in df.columns:
            return {'min': 0, 'max': 100}
        
        return {
            'min': float(df[amount_col].min()),
            'max': float(df[amount_col].max())
        }
    
    def get_date_range(self, df: pd.DataFrame,
                      date_col: str = 'date') -> Dict[str, datetime]:
        """Get min and max dates for date filter.
        
        Args:
            df: DataFrame containing date data
            date_col: Name of date column
            
        Returns:
            Dictionary with min and max dates
        """
        if df.empty or date_col not in df.columns:
            return {
                'min': datetime.now() - timedelta(days=30),
                'max': datetime.now()
            }
        
        df_dates = pd.to_datetime(df[date_col])
        return {
            'min': df_dates.min().to_pydatetime(),
            'max': df_dates.max().to_pydatetime()
        }
