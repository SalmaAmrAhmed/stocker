import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from typing import Dict, TypedDict, List, Union, Any
from langgraph.graph import StateGraph, END, START
import yfinance as yf
import pandas as pd
import traceback
import os
# from IPython.display import Image, display



class AgentState(TypedDict):
    """
    Holds the complete state of our analysis workflow:
    - messages: Communication history
    - screened_stocks: List of stocks that passed screening
    - stock_data: Detailed financial data for each stock
    - analysis: Technical analysis results
    - decisions: Trading decisions
    - debug_info: Debugging information
    """
    messages: List[Union[HumanMessage, AIMessage]]
    screened_stocks: List[str]
    stock_data: Dict[str, Dict[str, Any]]
    analysis: Dict[str, str]
    decisions: Dict[str, str]
    debug_info: Dict[str, Any]


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    RSI measures momentum by comparing the magnitude of recent gains to recent losses.
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        float: RSI value between 0 and 100
    """
    try:
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
        rs = gain / loss
        return float(100 - (100 / (1 + rs.iloc[-1])))
    
    except Exception:
        return 50.0
    

def calculate_atr(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for a given stock ticker.
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (int): Look-back period for ATR calculation (default: 14)
    start_date (str): Start date in 'YYYY-MM-DD' format (optional)
    end_date (str): End date in 'YYYY-MM-DD' format (optional)
    
    Returns:
    pd.Series: ATR values for the specified period
    """

    # Calculate True Range
    prices['High-Low'] = prices['High'] - prices['Low']
    prices['High-PrevClose'] = abs(prices['High'] - prices['Close'].shift(1))
    prices['Low-PrevClose'] = abs(prices['Low'] - prices['Close'].shift(1))
    
    prices['True_Range'] = prices[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    # Calculate ATR
    prices['ATR'] = prices['True_Range'].rolling(window=period).mean()
    
    return float(prices['ATR'].iloc[-1])

# def calculate_support_resistance(prices: pd.Series, window: int = 20) -> tuple[float, float]:
#     """
#     Calculate support and resistance levels using rolling min/max method.
    
#     Args:
#         prices: Series of closing prices
#         window: Period for calculation (default 20 days)
    
#     Returns:
#         tuple: (support_level, resistance_level)
#     """
#     try:
#         support = prices.rolling(window=window).min().iloc[-1]
#         resistance = prices.rolling(window=window).max().iloc[-1]
#         return float(support), float(resistance)
#     except Exception:
#         avg_price = prices.mean()
#         return float(avg_price), float(avg_price)


def get_basic_stock_info(symbol: str) -> dict:
    """Helper function to fetch basic stock information"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        info = stock.info
        
        if hist.empty:
            return None
            
        avg_volume = hist['Volume'].sum()
        current_price = hist['Close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'price': current_price,
            'volume': avg_volume,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'raw_info': info,
            'data_quality': {
                'has_price': not pd.isna(current_price),
                'has_volume': not pd.isna(avg_volume),
                'price_above_zero': current_price > 0,
                'volume_above_zero': avg_volume > 0
            }
        }
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {str(e)}")
        return None




def screen_stocks(state: AgentState) -> AgentState:
    """
    Screen stocks based on predefined criteria and collect initial data.
    Implements robust error handling and fallback mechanisms.
    """
    state['debug_info'] = {'screening_step': {}}
    
    stocks_processed = 0
    valid_stocks = []
    screening_results = {}

    for symbol in state['screened_stocks']:
        st.info(f"Processing {symbol}....")
        stocks_processed += 1

        stock_info = get_basic_stock_info(symbol)
        
        if stock_info:
            passed = True
            reasons = []
            
            if stock_info['price'] < 1:
                reasons.append(f"Price (${stock_info['price']:.2f}) below $1")
                passed = False
                
            if stock_info['volume'] < 100000:
                reasons.append(f"Volume ({stock_info['volume']:.0f}) below 100,000")
                passed = False
            
            screening_results[symbol] = {
                'sector': stock_info.get('sector', 'Unknown'),
                'passed': passed,
                'reasons': reasons,
                'data': stock_info
            }
            
            if passed:
                valid_stocks.append(symbol)
            else:
                st.warning(f"❌ {symbol} failed screening: {reasons}")

    if valid_stocks:
        valid_stocks.sort(
            key=lambda x: screening_results[x]['data']['volume'],
            reverse=True
        )
        state['screened_stocks'] = valid_stocks
        state['debug_info']['screening_step'] = {
            'method': 'enhanced_screening',
            'success': True,
            'stocks_found': len(valid_stocks),
            'total_processed': len(screening_results),
            'screening_results': screening_results
        }
    else:
        fallback_stocks = ['GOOGL']
        state['screened_stocks'] = fallback_stocks
        state['debug_info']['screening_step'] = {
            'method': 'fallback',
            'reason': 'no_valid_stocks_found',
            'screening_results': screening_results
        }

    return state

def fetch_stock_data(state: AgentState) -> AgentState:
    """
    Fetch comprehensive stock data and calculate technical indicators.
    Includes enhanced error handling and data validation.
    """
    state['debug_info']['yahoo_data'] = {}
    valid_stocks = []
    
    for symbol in state['screened_stocks']:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            
            if not hist.empty:
                atr_value = calculate_atr(hist)

                current_price = float(hist['Close'].iloc[-1])
                atr_percent = (atr_value / current_price) * 100
                
                # support, resistance = calculate_support_resistance(hist['Close'])
                
                stop_loss = current_price - (atr_value * 2)
                take_profit = current_price + (atr_value * 3)
                risk_reward = (take_profit - current_price) / (current_price - stop_loss) if stop_loss != current_price else 0
                
                stock_data = {
                    "current_price": current_price,
                    "previous_price": float(hist['Close'].iloc[-2]),
                    "volume": float(hist['Volume'].iloc[-1]),
                    "sma_20": float(hist['Close'].rolling(window=20).mean().iloc[-1]),
                    "sma_50": float(hist['Close'].rolling(window=50).mean().iloc[-1]),
                    "sma_200": float(hist['Close'].rolling(window=200).mean().iloc[-1]),
                    "price_change": float(((current_price - hist['Close'].iloc[-2]) / 
                                  hist['Close'].iloc[-2] * 100)),
                    "volume_change": float(((hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2]) /
                                   hist['Volume'].iloc[-2] * 100)),
                    "rsi": calculate_rsi(hist['Close']),
                    "atr": atr_value,
                    "atr_percent": atr_percent,
                    # "support": support,
                    # "resistance": resistance,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward": risk_reward,
                    "volatility_upper": current_price + (2 * atr_value),
                    "volatility_lower": current_price - (2 * atr_value)
                }
                
                state['stock_data'][symbol] = stock_data
                valid_stocks.append(symbol)
                state['debug_info']['yahoo_data'][symbol] = 'success'
                
            else:
                st.warning(f"No historical data available for {symbol}")
                state['debug_info']['yahoo_data'][symbol] = 'no_data'
                
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)}")
            state['debug_info']['yahoo_data'][symbol] = str(e)
    
    state['screened_stocks'] = valid_stocks
    
    if not valid_stocks:
        st.error("No valid stock data could be retrieved")
    else:
        st.success(f"Successfully retrieved data for {len(valid_stocks)} stocks")
    
    return state

def generate_insights(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Generate comprehensive technical analysis insights using all available indicators.
    Provides detailed analysis with specific trade setups and risk management guidance.
    """
    for symbol in state['screened_stocks']:
        try:
            stock_data = state['stock_data'][symbol]
            
            atr_percent = stock_data['atr_percent']
            volatility_state = "high" if atr_percent > 3 else "moderate" if atr_percent > 1.5 else "low"
            
            current_price = stock_data['current_price']
            # support = stock_data['support']
            # resistance = stock_data['resistance']
            
            # support_distance = ((current_price - support) / support) * 100
            # resistance_distance = ((resistance - current_price) / current_price) * 100
            
            # range_size = resistance - support
            # position_in_range = ((current_price - support) / range_size) * 100 if range_size > 0 else 50
            
            prompt = f"""
            Analyze the following technical indicators for {symbol}:

            Price Action:
            - Current Price: ${current_price:.2f}
            - Price Change: {stock_data['price_change']:.2f}%

            Volatility Analysis:
            - ATR: ${stock_data['atr']:.2f} ({stock_data['atr_percent']:.2f}% of price)
            - Volatility State: {volatility_state}
            - Volatility Bands: ${stock_data['volatility_lower']:.2f} - ${stock_data['volatility_upper']:.2f}
            - Volume: {stock_data['volume']:,.0f}
            - Volume Change: {stock_data['volume_change']:.2f}%

            Momentum and Trend:
            - RSI: {stock_data['rsi']:.2f}
            - 20-day SMA: ${stock_data['sma_20']:.2f}
            - 50-day SMA: ${stock_data['sma_50']:.2f}
            - 200-day SMA: ${stock_data['sma_200']:.2f}

            Risk Management:
            - Stop Loss: ${stock_data['stop_loss']:.2f}
            - Take Profit: ${stock_data['take_profit']:.2f}
            - Risk/Reward Ratio: {stock_data['risk_reward']:.2f}

            Provide a comprehensive technical analysis in 3-4 sentences focusing on:
            1. Current trend and price action relative to support/resistance levels
            2. Volatility analysis using ATR and volume patterns
            3. Momentum signals and potential trade setups with specific entry/exit points
            4. Risk management considerations based on ATR and volatility bands
            """

            st.info("Running AI Analysis ...")
            
            analysis = llm.invoke(prompt)
            state['analysis'][symbol] = analysis
            state['decisions'][symbol] = "PENDING"
            
            if 'analysis_metrics' not in state['debug_info']:
                state['debug_info']['analysis_metrics'] = {}
            
            state['debug_info']['analysis_metrics'][symbol] = {
                # 'support_distance': support_distance,
                # 'resistance_distance': resistance_distance,
                # 'position_in_range': position_in_range,
                'volatility_state': volatility_state
            }
            
        except Exception as e:
            st.warning(f"Analysis failed for {symbol}: {str(e)}")
            state['analysis'][symbol] = "Analysis failed"
            state['decisions'][symbol] = "ERROR"



def make_trading_decisions(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Fourth node: Generate specific trading decisions based on analysis.
    Includes improved error handling and response processing.
    
    The function takes the technical analysis for each stock and generates
    a clear trading decision using the LLM. It carefully processes the 
    LLM's response to ensure we get a valid decision category.
    """

    valid_decisions = {
        "STRONG BUY",
        "BUY",
        "HOLD",
        "SELL",
        "STRONG SELL"
    }
    

    decision_tracking = {}
    
    for symbol in state['screened_stocks']:
        
        if state['analysis'].get(symbol, "Analysis failed") == "Analysis failed":
            st.warning(f"Skipping decision generation for {symbol} due to failed analysis")
            state['decisions'][symbol] = "HOLD"
            decision_tracking[symbol] = {
                'status': 'skipped',
                'reason': 'analysis_failed'
            }
            continue
            
        try:
            analysis = state['analysis'][symbol]
            
            prompt = f"""
            Based on the technical analysis for {symbol}:
            {analysis}
            
            Analyze this information and respond with exactly ONE of these trading decisions:
            STRONG BUY
            BUY
            HOLD
            SELL
            STRONG SELL
            
            Respond with only one of these exact terms, nothing else.
            """
            
            response = llm.invoke(prompt)
            
            if isinstance(response, str):
                decision = response.strip().upper()
            else:
                decision = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            

            if decision in valid_decisions:
                state['decisions'][symbol] = decision
                decision_tracking[symbol] = {
                    'status': 'success',
                    'raw_response': response,
                    'processed_decision': decision
                }

            else:

                state['decisions'][symbol] = "HOLD"
                decision_tracking[symbol] = {
                    'status': 'invalid_decision',
                    'raw_response': response,
                    'fallback': 'HOLD'
                }
                
        except Exception as e:
            error_details = traceback.format_exc()
            st.warning(f"Error generating decision for {symbol}: {str(e)}")
            
            st.code(error_details, language='python')
            
            state['decisions'][symbol] = "HOLD"
            decision_tracking[symbol] = {
                'status': 'error',
                'error': str(e),
                'traceback': error_details,
                'fallback': 'HOLD'
            }
    
    
    state['debug_info']['decision_generation'] = decision_tracking
    
    total_stocks = len(state['screened_stocks'])
    successful_decisions = sum(1 for info in decision_tracking.values() if info['status'] == 'success')
    
    st.write(f"""
    Decision Generation Summary:
    - Total stocks processed: {total_stocks}
    - Successful decisions: {successful_decisions}
    - Fallback to HOLD: {total_stocks - successful_decisions}
    """)
    
    return state

def create_analysis_workflow(llm: ChatOpenAI) -> StateGraph:
    """
    Create and configure the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("screen_stocks", screen_stocks)
    workflow.add_node("fetch_stock_data", fetch_stock_data)
    workflow.add_node("generate_insights", lambda state: generate_insights(state, llm))
    workflow.add_node("make_trading_decisions", lambda state: make_trading_decisions(state, llm))
    
    workflow.add_edge(START, "screen_stocks")
    workflow.add_edge("screen_stocks", "fetch_stock_data")
    workflow.add_edge("fetch_stock_data", "generate_insights")
    workflow.add_edge("generate_insights", "make_trading_decisions")
    workflow.add_edge("make_trading_decisions", END)
    
    return workflow.compile()

def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Stock Gazing")
    api_key = os.getenv("OPENAI_API_KEY")
    
    with st.sidebar:
        st.header("Configuration")

        symbol = st.sidebar.text_input("Enter Stock Symbol:")
        
        if st.button("Run Analysis"):
            
            try:
                llm = ChatOpenAI(model="gpt-4-turbo", api_key=api_key)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                workflow = create_analysis_workflow(llm)

                # st.image(display(Image(workflow.get_graph().draw_mermaid_png())))
                
                initial_state = AgentState(
                    messages=[],
                    screened_stocks=[symbol],
                    stock_data={},
                    analysis={},
                    decisions={},
                    debug_info={}
                )
                
                status_text.text("Running analysis workflow...")
                final_state = workflow.invoke(initial_state)
                
                if not final_state['screened_stocks']:
                    st.error("No valid stocks found for analysis")
                    return
                
                st.session_state.final_state = final_state
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return

    if hasattr(st.session_state, 'final_state'):
        state = st.session_state.final_state
        
        if state['screened_stocks']:
            results_data = []
            for symbol in state['screened_stocks']:
                if symbol in state['stock_data']:
                    stock_data = state['stock_data'][symbol]
                    
                    analysis = state['analysis'].get(symbol, "N/A")
                    if hasattr(analysis, 'content'):
                        analysis = analysis.content
                    
                    
                    decision = state['decisions'].get(symbol, "N/A")
                    if hasattr(decision, 'content'):
                        decision = decision.content

                    results_data.append({
                        'Symbol': symbol,
                        'Price': f"${stock_data['current_price']:.2f}",
                        'Change %': f"{stock_data['price_change']:.2f}%",
                        'RSI': f"{stock_data['rsi']:.1f}",
                        'Volume': f"{stock_data['volume']:,.0f}",
                        # 'Support': f"${stock_data['support']:.2f}",
                        # 'Resistance': f"${stock_data['resistance']:.2f}",
                        'ATR': f"${stock_data['atr']:.2f}",
                        'Stop Loss': f"${stock_data['stop_loss']:.2f}",
                        'Take Profit': f"${stock_data['take_profit']:.2f}",
                        'Risk/Reward': f"{stock_data['risk_reward']:.2f}",
                        'Analysis': str(analysis),
                        'Trading Decision': str(decision).strip()
                    })
            
            if results_data:
                df = pd.DataFrame(results_data)
                
                # Transpose the first row of data (since we're only analyzing one stock at a time)
                if len(df) > 0:
                    transposed_df = pd.DataFrame(df.iloc[0]).reset_index()
                    transposed_df.columns = ['Metric', 'Value']
                    
                    # Apply styling using st.table with custom HTML
                    def style_row(row):
                        if row['Metric'] == 'Trading Decision':
                            decision = str(row['Value']).upper()
                            if 'STRONG BUY' in decision:
                                return ['background-color: darkgreen; color: white'] * len(row)
                            elif 'BUY' in decision:
                                return ['background-color: lightgreen'] * len(row)
                            elif 'SELL' in decision:
                                return ['background-color: salmon'] * len(row)
                            elif 'STRONG SELL' in decision:
                                return ['background-color: darkred; color: white'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = transposed_df.style.apply(style_row, axis=1)
                    st.table(styled_df)
                
                # Keep the download button
                # csv = df.to_csv(index=False)
                # st.download_button(
                #     label="Download Analysis Results",
                #     data=csv,
                #     file_name="stock_analysis.csv",
                #     mime="text/csv"
                # )
            else:
                st.warning("No valid results to display")
        else:
            st.warning("No stocks found matching the criteria")


if __name__ == "__main__":
    main()

        #analysis
        #macd
        #japanese candlesticks

    #strategies
        #trade the range (use wiliams indicator with duration of 14 days) / #trade the support and resistance (use support and resistance strategy) 
        #trade the breakout (use breakout strategy) enter on breaking resistance point