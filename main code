import requests
import yfinance as yf
from datetime import datetime, timedelta
import json

# API Configuration

url = "https://api.perplexity.ai/chat/completions"

def get_company_info(ticker_symbol):
    """
    Get company sector, industry, and basic information using yfinance
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        company_data = {
            'name': info.get('longName', ticker_symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 'Unknown'),
            'current_price': info.get('currentPrice', 'Unknown')
        }
        
        return company_data
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return None

def analyze_company_news(company_data):
    """
    Use Perplexity API to analyze company news, sector trends, and stock performance
    """
    
    # Construct comprehensive prompt for news analysis
    prompt = f"""
    Analyze the following company and provide comprehensive news insights:

    Company: {company_data['name']} ({company_data['sector']} sector, {company_data['industry']} industry)

    Please provide:

    1. RECENT NEWS (Past 2 weeks):
    - Company-specific news (excluding price movements and product/service promotions)
    - Sector and industry news that could impact this company
    - Focus on regulatory changes, market trends, competitive developments

    2. NOTABLE STOCK MOVEMENTS (Past year):
    - News events that caused stock price changes of 5% or more
    - Include both positive and negative catalysts
    - Explain the underlying reasons for these movements

    3. THREE-SENTENCE SUMMARY:
    - Concise overview of all recent developments
    - Focus on most impactful news items

    4. OVERARCHING STOCK ANALYSIS:
    - How recent news affects the company's outlook
    - Risk factors and opportunities identified
    - Overall assessment of the stock's current position

    Format your response clearly with these four sections. Be specific and avoid generic statements.
    """
    
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar-reasoning",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,  # Increased token limit for comprehensive analysis
        "temperature": 0.3    # Lower temperature for more focused analysis
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            analysis = result['choices'][0]['message']['content']
            return analysis
        else:
            return "Error: Unable to get analysis from Perplexity API"
            
    except requests.exceptions.RequestException as e:
        return f"API request error: {e}"
    except json.JSONDecodeError as e:
        return f"JSON parsing error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

def get_stock_performance(ticker_symbol, days=365):
    """
    Get stock performance data for the past year
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
            
        # Calculate key metrics
        current_price = hist['Close'].iloc[-1]
        start_price = hist['Close'].iloc[0]
        total_return = ((current_price - start_price) / start_price) * 100
        
        # Find significant price movements (>5%)
        significant_movements = []
        for i in range(1, len(hist)):
            daily_return = ((hist['Close'].iloc[i] - hist['Close'].iloc[i-1]) / hist['Close'].iloc[i-1]) * 100
            if abs(daily_return) >= 5:
                significant_movements.append({
                    'date': hist.index[i].strftime('%Y-%m-%d'),
                    'return': daily_return,
                    'price': hist['Close'].iloc[i]
                })
        
        performance_data = {
            'current_price': current_price,
            'start_price': start_price,
            'total_return': total_return,
            'significant_movements': significant_movements,
            'volatility': hist['Close'].pct_change().std() * 100
        }
        
        return performance_data
        
    except Exception as e:
        print(f"Error fetching stock performance: {e}")
        return None

def comprehensive_company_analysis(ticker_symbol):
    """
    Main function to perform comprehensive company analysis
    """
    print(f"Starting comprehensive analysis for {ticker_symbol}...")
    
    # Get company information
    company_data = get_company_info(ticker_symbol)
    if not company_data:
        return "Failed to get company information"
    
    print(f"Company: {company_data['name']}")
    print(f"Sector: {company_data['sector']}")
    print(f"Industry: {company_data['industry']}")
    print(f"Current Price: ${company_data['current_price']}")
    print("-" * 50)
    
    # Get stock performance data
    performance_data = get_stock_performance(ticker_symbol)
    if performance_data:
        print(f"1-Year Return: {performance_data['total_return']:.2f}%")
        print(f"Volatility: {performance_data['volatility']:.2f}%")
        print(f"Significant movements (>5%): {len(performance_data['significant_movements'])}")
        print("-" * 50)
    
    # Get news analysis from Perplexity
    print("Analyzing news and market developments...")
    news_analysis = analyze_company_news(company_data)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE NEWS ANALYSIS")
    print("="*60)
    print(news_analysis)
    
    return {
        'company_data': company_data,
        'performance_data': performance_data,
        'news_analysis': news_analysis
    }

# Example usage
if __name__ == "__main__":
    # Get user input for ticker symbol
    print("="*60)
    print("COMPANY NEWS ANALYSIS SYSTEM")
    print("="*60)
    print("This system analyzes company news, sector trends, and stock performance")
    print("using Perplexity AI and yfinance data.")
    print("="*60)
    
    while True:
        ticker = input("\nEnter the ticker symbol to analyze (or 'quit' to exit): ").strip().upper()
        
        if ticker.lower() == 'quit':
            print("Exiting analysis system. Goodbye!")
            break
            
        if not ticker:
            print("Please enter a valid ticker symbol.")
            continue
            
        print(f"\nAnalyzing {ticker}...")
        
        try:
            result = comprehensive_company_analysis(ticker)
            
            if isinstance(result, dict):
                print("\n✅ Analysis completed successfully!")
                print(f"Company: {result['company_data']['name']}")
                print(f"Analysis length: {len(result['news_analysis'])} characters")
                print("\n" + "="*60)
                print("ANALYSIS COMPLETE - Review the results above")
                print("="*60)
            else:
                print(f"❌ Analysis failed: {result}")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            print("Please check your internet connection and API key configuration.")
        
        # Ask if user wants to analyze another company
        another = input("\nWould you like to analyze another company? (y/n): ").strip().lower()
        if another not in ['y', 'yes']:
            print("Exiting analysis system. Goodbye!")
            break
