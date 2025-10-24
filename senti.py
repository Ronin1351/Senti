import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any
import json

class GlassmorphismSentimentUI:
    def __init__(self):
        self.setup_page_config()
        self.setup_css()
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="Gold Sentiment Dashboard",
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
    def setup_css(self):
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        }
        
        .neon-border {
            border: 1px solid;
            border-image: linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe) 1;
            animation: borderGlow 2s ease-in-out infinite alternate;
        }
        
        @keyframes borderGlow {
            from { box-shadow: 0 0 20px rgba(0, 242, 254, 0.3); }
            to { box-shadow: 0 0 30px rgba(0, 242, 254, 0.6), 0 0 40px rgba(79, 172, 254, 0.4); }
        }
        
        .signal-positive {
            background: linear-gradient(135deg, rgba(0, 255, 127, 0.2), rgba(0, 255, 127, 0.1));
            border-left: 4px solid #00ff7f;
        }
        
        .signal-negative {
            background: linear-gradient(135deg, rgba(255, 0, 128, 0.2), rgba(255, 0, 128, 0.1));
            border-left: 4px solid #ff0080;
        }
        
        .signal-neutral {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 215, 0, 0.1));
            border-left: 4px solid #ffd700;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.8);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        .sentiment-badge {
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }
        
        .bullish { background: rgba(0, 255, 127, 0.2); color: #00ff7f; border: 1px solid rgba(0, 255, 127, 0.3); }
        .bearish { background: rgba(255, 0, 128, 0.2); color: #ff0080; border: 1px solid rgba(255, 0, 128, 0.3); }
        .neutral { background: rgba(255, 215, 0, 0.2); color: #ffd700; border: 1px solid rgba(255, 215, 0, 0.3); }
        
        .article-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 15px;
            margin: 8px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.2s ease;
        }
        
        .article-card:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        
        .driver-tag {
            display: inline-block;
            padding: 2px 8px;
            margin: 2px;
            border-radius: 10px;
            font-size: 0.7rem;
            background: rgba(79, 172, 254, 0.2);
            color: #4facfe;
            border: 1px solid rgba(79, 172, 254, 0.3);
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00f2fe, #4facfe);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_header(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='color: white; font-size: 3rem; font-weight: 700; margin: 0; 
                    background: linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-shadow: 0 0 30px rgba(0, 242, 254, 0.5);'>
                    GOLD SENTIMENT DASHBOARD
                </h1>
                <p style='color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin: 0;'>
                    Real-time News Sentiment Analysis & Trading Signals
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def create_signal_card(self, signal_data: Dict[str, Any]):
        """Create main signal card with neon glow effects"""
        signal = signal_data.get('intraday_signal', {})
        swing = signal_data.get('swing_overlay', {})
        
        # Determine signal styling
        if signal.get('signal', 0) > 0:
            signal_class = "signal-positive"
            signal_text = "BULLISH"
            signal_emoji = "📈"
        elif signal.get('signal', 0) < 0:
            signal_class = "signal-negative"
            signal_text = "BEARISH"
            signal_emoji = "📉"
        else:
            signal_class = "signal-neutral"
            signal_text = "NEUTRAL"
            signal_emoji = "➡️"
        
        st.markdown(f"""
        <div class='glass-card neon-border {signal_class}' style='margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <div class='metric-label'>INTRADAY SIGNAL</div>
                    <div class='metric-value'>{signal_text} {signal_emoji}</div>
                    <div style='color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;'>
                        Z-Score: {signal.get('z', 0)} | Strength: {signal.get('strength', 0)} | Articles: {signal.get('count', 0)}
                    </div>
                </div>
                <div style='text-align: right;'>
                    <div class='metric-label'>SWING OVERLAY</div>
                    <div style='font-size: 1.5rem; font-weight: 600; color: {"#00ff7f" if swing.get("filter", 0) > 0 else "#ff0080" if swing.get("filter", 0) < 0 else "#ffd700"}'>
                        {swing.get("filter", 0)}
                    </div>
                    <div style='color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;'>
                        Size Multiplier: {swing.get('size_mult', 1.0)}x
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_metrics_row(self, sentiment_data: Dict[str, Any]):
        """Create metrics cards row"""
        col1, col2, col3, col4 = st.columns(4)
        
        articles = sentiment_data.get('latest_articles', [])
        avg_sentiment = np.mean([art.get('sent', 0) for art in articles]) if articles else 0
        avg_driver_bias = np.mean([art.get('drv', 0) for art in articles]) if articles else 0
        
        metrics = [
            ("TOTAL ARTICLES", len(articles), "#00f2fe"),
            ("AVG SENTIMENT", f"{avg_sentiment:.3f}", "#4facfe"),
            ("DRIVER BIAS", f"{avg_driver_bias:.3f}", "#00ff7f"),
            ("SIGNAL STRENGTH", sentiment_data.get('intraday_signal', {}).get('strength', 0), "#ff0080")
        ]
        
        for col, (label, value, color) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='background: linear-gradient(45deg, {color}, {color}80); -webkit-background-clip: text;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def create_sentiment_chart(self, articles: List[Dict]):
        """Create sentiment distribution chart"""
        if not articles:
            return
            
        sentiments = [art.get('sent', 0) for art in articles]
        times = [art.get('ts', '') for art in articles]
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=sentiments,
            nbinsx=20,
            marker_color='#00f2fe',
            opacity=0.7,
            name='Sentiment Distribution'
        ))
        
        fig.update_layout(
            title="Sentiment Distribution",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        fig.update_xaxes(title_text="Sentiment Score", gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(title_text="Frequency", gridcolor='rgba(255,255,255,0.1)')
        
        return fig
    
    def create_timeseries_chart(self, series_data: List[Dict]):
        """Create time series chart for sentiment scores"""
        if not series_data:
            return
            
        df = pd.DataFrame(series_data)
        if 'bucket' in df.columns and 'net_score' in df.columns:
            df['bucket'] = pd.to_datetime(df['bucket'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['bucket'],
                y=df['net_score'],
                mode='lines+markers',
                line=dict(color='#00f2fe', width=3),
                marker=dict(size=6, color='#4facfe'),
                name='Net Sentiment'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
            
            fig.update_layout(
                title="Sentiment Time Series",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            fig.update_xaxes(title_text="Time", gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(title_text="Net Score", gridcolor='rgba(255,255,255,0.1)')
            
            return fig
    
    def create_articles_list(self, articles: List[Dict]):
        """Create articles list with sentiment badges and clickable cards"""
        if not articles:
            return

        for i, article in enumerate(articles[:10]):  # Show top 10
            sentiment = article.get('sent', 0)
            sentiment_class = "bullish" if sentiment > 0.1 else "bearish" if sentiment < -0.1 else "neutral"
            sentiment_text = "BULLISH" if sentiment > 0.1 else "BEARISH" if sentiment < -0.1 else "NEUTRAL"

            # Extract drivers
            drivers = article.get('flags', {})
            active_drivers = [k for k, v in drivers.items() if v > 0]

            # Create expander for each article
            with st.expander(f"📄 {article.get('t', 'No title')}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"""
                    <div style='color: rgba(255, 255, 255, 0.9); margin-bottom: 10px;'>
                        <strong>Source:</strong> {article.get('src', 'Unknown')}<br>
                        <strong>Published:</strong> {article.get('ts', '')[:19].replace('T', ' ')}<br>
                        <strong>Sentiment Score:</strong> <span style='color: {"#00ff7f" if sentiment > 0 else "#ff0080" if sentiment < 0 else "#ffd700"}'>{sentiment:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display article content
                    if article.get('content'):
                        st.markdown(f"""
                        <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px;
                                    border-left: 3px solid {"#00ff7f" if sentiment > 0 else "#ff0080" if sentiment < 0 else "#ffd700"};
                                    color: rgba(255, 255, 255, 0.8); line-height: 1.6;'>
                            {article.get('content')}
                        </div>
                        """, unsafe_allow_html=True)

                    # Link to source
                    if article.get('url'):
                        st.markdown(f"""
                        <div style='margin-top: 10px;'>
                            <a href='{article.get('url')}' target='_blank'
                               style='color: #4facfe; text-decoration: none; font-size: 0.9rem;'>
                                🔗 Read full article on {article.get('src', 'source')}
                            </a>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class='sentiment-badge {sentiment_class}'>{sentiment_text}</span>
                    </div>
                    <div style='margin-top: 15px; padding: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 10px;'>
                        <div style='color: rgba(255, 255, 255, 0.7); font-size: 0.75rem; margin-bottom: 5px;'>METRICS</div>
                        <div style='color: white; font-size: 0.8rem;'>
                            <strong>Driver:</strong> {article.get('drv', 0):.3f}<br>
                            <strong>Relevance:</strong> {article.get('rel', 0):.2f}<br>
                            <strong>Credibility:</strong> {article.get('cred', 0):.2f}<br>
                            <strong>Recency:</strong> {article.get('rec', 0):.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Display active drivers as tags
                if active_drivers:
                    st.markdown("<div style='margin-top: 10px;'><strong style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>ACTIVE DRIVERS:</strong></div>", unsafe_allow_html=True)
                    st.markdown(
                        '<div style="margin-top: 5px;">' +
                        ''.join([f'<span class="driver-tag">{driver.upper().replace("_", " ")}</span>' for driver in active_drivers]) +
                        '</div>',
                        unsafe_allow_html=True
                    )
    
    def create_drivers_breakdown(self, articles: List[Dict]):
        """Create driver frequency breakdown"""
        if not articles:
            return
            
        driver_counts = {}
        for article in articles:
            drivers = article.get('flags', {})
            for driver, active in drivers.items():
                if active:
                    driver_counts[driver] = driver_counts.get(driver, 0) + 1
        
        if driver_counts:
            drivers_df = pd.DataFrame({
                'Driver': list(driver_counts.keys()),
                'Count': list(driver_counts.values())
            }).sort_values('Count', ascending=True)
            
            fig = go.Figure(go.Bar(
                y=drivers_df['Driver'],
                x=drivers_df['Count'],
                orientation='h',
                marker_color='#00f2fe',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Driver Frequency",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            fig.update_xaxes(title_text="Frequency", gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(title_text="", gridcolor='rgba(255,255,255,0.1)')
            
            return fig

    def render_dashboard(self, sentiment_data: Dict[str, Any]):
        """Main dashboard rendering function"""
        self.create_header()
        
        # Main signal row
        self.create_signal_card(sentiment_data)
        
        # Metrics row
        self.create_metrics_row(sentiment_data)
        
        # Charts and articles
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Charts row
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                sentiment_chart = self.create_sentiment_chart(sentiment_data.get('latest_articles', []))
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
            
            with chart_col2:
                timeseries_chart = self.create_timeseries_chart(sentiment_data.get('series_tail', []))
                if timeseries_chart:
                    st.plotly_chart(timeseries_chart, use_container_width=True)
            
            # Drivers breakdown
            drivers_chart = self.create_drivers_breakdown(sentiment_data.get('latest_articles', []))
            if drivers_chart:
                st.plotly_chart(drivers_chart, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class='glass-card' style='margin-bottom: 20px;'>
                <h3 style='color: white; margin-bottom: 15px;'>📰 LATEST ARTICLES</h3>
            </div>
            """, unsafe_allow_html=True)
            
            self.create_articles_list(sentiment_data.get('latest_articles', []))

# Sample data generator for demonstration
def generate_sample_data(seed=None):
    """Generate sample sentiment data for demonstration

    Args:
        seed: Optional random seed for reproducible data
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample article data with more variety
    sample_titles = [
        "Gold prices surge on inflation concerns and Fed policy speculation",
        "Central banks increase gold reserves amid economic uncertainty",
        "Dollar weakness drives gold to multi-month highs",
        "Treasury yields decline, boosting gold appeal",
        "Geopolitical tensions fuel safe-haven demand for gold",
        "Gold miners report strong quarterly earnings",
        "ETF inflows signal renewed investor interest in gold",
        "Gold pulls back as risk appetite returns to markets"
    ]

    sample_sources = [
        {'name': 'reuters.com', 'url': 'https://www.reuters.com/markets/commodities'},
        {'name': 'bloomberg.com', 'url': 'https://www.bloomberg.com/markets/commodities'},
        {'name': 'ft.com', 'url': 'https://www.ft.com/commodities'},
        {'name': 'wsj.com', 'url': 'https://www.wsj.com/market-data/commodities'}
    ]

    articles = []
    for i in range(8):
        sentiment = np.random.uniform(-0.8, 0.8)
        source = np.random.choice(sample_sources)

        # Generate sample article content
        content_snippets = [
            f"Gold {'rallied' if sentiment > 0 else 'declined'} in trading today as investors reacted to {'positive' if sentiment > 0 else 'negative'} economic indicators.",
            f"Market analysts suggest that {'inflation fears' if sentiment > 0 else 'rising interest rates'} are the primary driver.",
            f"The precious metal is {'benefiting from' if sentiment > 0 else 'pressured by'} recent dollar movements.",
            "Trading volumes have been notably high across major exchanges.",
            f"Technical indicators suggest {'further upside potential' if sentiment > 0 else 'continued weakness'} in the near term."
        ]

        articles.append({
            't': sample_titles[i] if i < len(sample_titles) else f"Gold prices {'rise' if sentiment > 0 else 'fall'} on market developments",
            'src': source['name'],
            'url': source['url'],  # Added URL field
            'content': ' '.join(np.random.choice(content_snippets, size=3, replace=False)),  # Added full content
            'ts': (datetime.now() - timedelta(hours=i, minutes=np.random.randint(0, 60))).isoformat(),
            'sent': round(sentiment, 3),
            'drv': round(np.random.uniform(-0.5, 0.5), 3),
            'rel': round(np.random.uniform(0.5, 1.0), 2),
            'cred': round(np.random.uniform(0.7, 1.0), 2),
            'rec': round(np.random.uniform(0.3, 1.0), 2),
            'score': round(sentiment * np.random.uniform(0.5, 1.0), 3),
            'flags': {
                'usd_down': np.random.choice([0, 1], p=[0.7, 0.3]),
                'yields_up': np.random.choice([0, 1], p=[0.8, 0.2]),
                'risk_off': np.random.choice([0, 1], p=[0.6, 0.4]),
                'inflation': np.random.choice([0, 1], p=[0.6, 0.4]),
                'fed_policy': np.random.choice([0, 1], p=[0.7, 0.3])
            }
        })

    series_data = []
    for i in range(10):
        series_data.append({
            'bucket': (datetime.now() - timedelta(hours=9-i)).isoformat(),
            'net_score': round(np.random.uniform(-0.5, 0.5), 3),
            'strength': round(np.random.uniform(0.5, 2.5), 2),
            'count': np.random.randint(1, 8)
        })

    signal_value = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

    return {
        'asset': 'gold',
        'timestamp': datetime.now().isoformat(),  # Added timestamp
        'intraday_signal': {
            'signal': signal_value,
            'z': round(np.random.uniform(-2, 2), 2),
            'strength': round(np.random.uniform(0.5, 3.0), 2),
            'count': len(articles)
        },
        'swing_overlay': {
            'filter': np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]),
            'size_mult': round(np.random.uniform(0.8, 1.2), 2)
        },
        'latest_articles': articles,
        'series_tail': series_data
    }

# Main application
def main():
    ui = GlassmorphismSentimentUI()

    # Initialize session state for data persistence
    if 'sentiment_data' not in st.session_state:
        # Generate initial data with a fixed seed for consistency
        st.session_state.sentiment_data = generate_sample_data(seed=42)
        st.session_state.data_loaded_at = datetime.now()

    # Get data from session state
    sentiment_data = st.session_state.sentiment_data

    # Display last update time
    st.markdown(f"""
    <div style='text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 0.85rem; margin-bottom: 10px;'>
        Last Updated: {st.session_state.data_loaded_at.strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

    # Render the dashboard
    ui.render_dashboard(sentiment_data)

    # Control buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("🔄 Refresh Data", use_container_width=True, help="Generate new sentiment data"):
                # Generate new data with a new random seed
                st.session_state.sentiment_data = generate_sample_data(seed=None)
                st.session_state.data_loaded_at = datetime.now()
                st.rerun()

        with button_col2:
            if st.button("📌 Keep Current", use_container_width=True, help="Keep displaying current data"):
                # Just rerun without changing data
                st.rerun()

    # Data info section
    with st.expander("ℹ️ About This Data", expanded=False):
        st.markdown("""
        ### Data Persistence

        **Fixed Issue:** The dashboard now uses session state to maintain consistent data across page refreshes.

        - **Browser Refresh:** Data stays the same (no random changes)
        - **Refresh Data Button:** Generates new sample data
        - **Keep Current Button:** Reloads the page with existing data

        ### Article Features

        Click on any article card to:
        - View full article content
        - See detailed sentiment metrics
        - Access the source URL
        - Review active market drivers

        ### For Production Use

        Replace `generate_sample_data()` with your actual sentiment data source:

        ```python
        # In main():
        if 'sentiment_data' not in st.session_state:
            from your_sentiment_engine import get_sentiment_data
            st.session_state.sentiment_data = get_sentiment_data('gold')
        ```
        """)

if __name__ == "__main__":
    main()