# NAVA Robo Advisor

NAVA Robo Advisor is a Flask-based web application that helps users simulate and optimize a portfolio of environmentally responsible ETFs using different investment strategies.

## Features

* Strategy-based Portfolio Optimization:

  * Minimum Variance (Low Risk)
  * Maximum Expected Return (High Risk)
  * Maximum Sharpe Ratio (Balanced)

* Input Options:

  * Expected Return sliders
  * Volatility caps
  * Custom or benchmark-based risk parameters

* Visual Outputs:

  * μ–σ Diagram
  * Smoothed ETF Weights Over Time
  * Cumulative Returns
  * Strategy Return Comparison
  * Daily Log Returns & Asset Prices (More Charts Page)

* User Management:

  * Registration, Login, Logout
  * Guest access
  * History of past simulations for authenticated users

<!-- * Download Report:

  * Generates PDF snapshot of key results and charts -->

* Admin Panel:

  * Flask-Admin secured by email authentication
  * User & History management

## How It Works

1. User selects a strategy and date range
2. App downloads price data from Yahoo Finance for selected ETFs
3. Performs optimization based on:

   * Log returns
   * Covariance matrix
   * Strategy objective
4. Visualizes results interactively
5. Saves results to database if user is logged in

## Project Structure

```
NAVARoboAdvisor/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   ├── forms.py
│   ├── robo/               # Optimization logic (RoboAdvisor class)
│   ├── templates/
│   └── static/
├── NAVA_ETF_Selection.xlsx
├── run.py
└── README.md
```

## Tech Stack

* Python + Flask
* Flask-SQLAlchemy + SQLite
* Flask-Login + Flask-WTF
* Plotly.js for charts
<!-- * HTML2PDF.js for client-side PDF -->
* Yahoo Finance (via yfinance)

## Installation

```bash
git clone https://github.com/yourusername/NAVA-robo-advisor
cd NAVA-robo-advisor
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Admin Access

To access `/admin`:

* Login with email: `admin@nava.com`
* Set password manually via shell if needed

## Future Enhancements

* Deployment on Render/Vercel/Fly.io
* User-specific reporting dashboard
* More ESG strategies (Carbon-neutral etc)

---

## About Us*


> **About NAVA Robo Advisor**
>
> This project was developed as part of a university seminar on Big Data and Sustainable Finance. It aims to empower users to build environmentally responsible portfolios through transparent, data-driven methods.
>
> The development was led by students of master's in business intelligence & data science at ISM Dortmund.