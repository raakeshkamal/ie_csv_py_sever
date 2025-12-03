def calculate_rebalancing(new_capital: float, current_values: dict, target_allocations: dict) -> dict:
    """
    Calculate rebalancing investments.
    
    Args:
        new_capital: Amount of new investment
        current_values: {ticker: current_value}
        target_allocations: {ticker: target_percentage}
    
    Returns:
        {
            'investments': [{'ticker': str, 'current_value': float, 'target_value': float, 'investment_amount': float}],
            'summary': {'total_current': float, 'new_total': float, 'total_investment': float}
        }
    """
    import numpy as np
    
    # Filter to common tickers
    common_tickers = set(current_values.keys()) & set(target_allocations.keys())
    
    # Normalize target allocations to sum to 100%
    total_target_pct = sum(target_allocations[t] for t in common_tickers)
    if total_target_pct == 0:
        raise ValueError("Target allocations sum to zero")
    
    normalized_targets = {t: (target_allocations[t] / total_target_pct) * 100 for t in common_tickers}
    
    # Current total value
    total_current = sum(current_values[t] for t in common_tickers)
    
    # New total portfolio value
    new_total = total_current + new_capital
    
    # Calculate target values and investments
    investments = []
    total_investment = 0.0
    for ticker in common_tickers:
        current_val = current_values.get(ticker, 0.0)
        target_pct = normalized_targets[ticker]
        target_val = new_total * (target_pct / 100.0)
        investment = max(0.0, target_val - current_val)
        investments.append({
            'ticker': ticker,
            'current_value': current_val,
            'target_value': target_val,
            'investment_amount': investment
        })
        total_investment += investment
    
    # Summary
    summary = {
        'total_current': total_current,
        'new_total': new_total,
        'total_investment': total_investment
    }
    
    return {
        'investments': investments,
        'summary': summary
    }
