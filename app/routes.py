from flask import Blueprint, render_template, url_for, flash, redirect, request
from flask_login import login_user, logout_user, current_user, login_required
from app import db
from app.models import User
from app.forms import RegistrationForm, LoginForm
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import History
from flask_login import login_required

import pandas as pd
from app.robo.Green_Robo_Advisor_Class import RoboAdvisor
import yfinance as yfin

bp = Blueprint('routes', __name__)

@bp.route('/')
def home():
    return render_template('index.html')

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pw = generate_password_hash(form.password.data)
        user = User(username=form.username.data, email=form.email.data, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('routes.login'))
    return render_template('register.html', title='Register', form=form)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('routes.home'))
        else:
            flash('Login failed. Please check email and password.', 'danger')
    return render_template('login.html', title='Login', form=form)

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('routes.home'))


@bp.route('/form')
def form():
    return render_template('form.html')

# @bp.route('/results', methods=['POST'])
# def results():
#     import numpy as np

#     strategy = request.form.get('strategy')
#     name = request.form.get('name')
#     start_date = request.form.get('start_date')
#     end_date = request.form.get('end_date')

#     expected_return = None
#     volatility_cap = None

#     if strategy == 'min-var':
#         option = request.form.get('expected_return_option')
#         if option == 'custom':
#             expected_return = float(request.form.get('custom_expected', 3)) / 100
#         elif option == 'low':
#             expected_return = 0.004
#         elif option == 'medium':
#             expected_return = 0.006
#         elif option == 'high':
#             expected_return = 0.008
#         expected_return = expected_return or 0.006

#     elif strategy == 'max-exp':
#         option = request.form.get('volatility_cap_option')
#         if option == 'custom':
#             volatility_cap = float(request.form.get('custom_volatility', 7)) / 100
#         elif option == 'low':
#             volatility_cap = max(0.006 - 2 * 0.002, 0.0)
#         elif option == 'medium':
#             volatility_cap = 0.006
#         elif option == 'high':
#             volatility_cap = 0.006 + 2 * 0.002
#         volatility_cap = volatility_cap or 0.006

#     ticker = pd.read_excel("Green_ETF_Selection.xlsx", sheet_name="ETF_Universe", engine="openpyxl")
#     tickers = list(ticker.Ticker)
#     labels = list(ticker.Label)
#     df = yfin.download(tickers, start=start_date, end=end_date)['Close']
#     df.columns = labels
#     df_cleaned = df.ffill().bfill()

#     RA = RoboAdvisor(df_cleaned, rf='Green Bonds', benchmark='MSCI World SRI')
#     logR = RA.getLogReturn(df_cleaned)

#     sol = RA.optimizeWeights(
#         logR=logR,
#         strategy=strategy,
#         mup=expected_return if expected_return is not None else 0.006,
#         sigmap=volatility_cap if volatility_cap is not None else 0.006,
#         printSol=False
#     )
    
#     # Save charts to static folder
#     RA.save_all_charts()

#     mu = logR.mean()
#     Cov = logR.cov()
#     rf = RA.rf
#     sharpe_ratio = round(((sol @ mu.T - rf) / (sol @ Cov @ sol.T) ** 0.5), 4)

#     # Compute histogram of daily log returns
#     total_returns = logR.sum(axis=1)
#     hist, bin_edges = np.histogram(total_returns, bins=10)
#     bin_centers = [round((bin_edges[i] + bin_edges[i+1]) / 2, 4) for i in range(len(bin_edges) - 1)]

#     if current_user.is_authenticated:
#         from app.models import History
#         new_entry = History(
#             user_id=current_user.id,
#             strategy=strategy,
#             expected_return=expected_return,
#             volatility_cap=volatility_cap,
#             result_summary=f"Sharpe Ratio: {sharpe_ratio}"
#         )
#         db.session.add(new_entry)
#         db.session.commit()

#     return render_template('results.html',
#                            strategy=strategy,
#                            expected_return=round((expected_return or 0.0) * 100, 2),
#                            volatility_cap=round((volatility_cap or 0.0) * 100, 2),
#                            sharpe_ratio=sharpe_ratio,
#                            labels=list(df_cleaned.columns),
#                            weights=[round(w, 4) for w in sol],
#                            log_bins=bin_centers,
#                            log_counts=hist.tolist())


@bp.route('/history')
@login_required
def history():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('history.html', history=user_history)

@bp.route('/more-charts')
def more_charts():
    return render_template('more_charts.html')

@bp.route('/results', methods=['POST'])
def results():
    import pandas as pd
    import yfinance as yfin
    from app.robo.Green_Robo_Advisor_Class import RoboAdvisor
    import numpy as np

    strategy = request.form.get('strategy')
    name = request.form.get('name')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    expected_return = None
    volatility_cap = None

    if strategy == 'min-var':
        option = request.form.get('expected_return_option')
        if option == 'custom':
            expected_return = float(request.form.get('custom_expected', 3)) / 100
        elif option == 'low':
            expected_return = 0.004
        elif option == 'medium':
            expected_return = 0.006
        elif option == 'high':
            expected_return = 0.008
        expected_return = expected_return or 0.006

    elif strategy == 'max-exp':
        option = request.form.get('volatility_cap_option')
        if option == 'custom':
            volatility_cap = float(request.form.get('custom_volatility', 7)) / 100
        elif option == 'low':
            volatility_cap = max(0.006 - 2 * 0.002, 0.0)
        elif option == 'medium':
            volatility_cap = 0.006
        elif option == 'high':
            volatility_cap = 0.006 + 2 * 0.002
        volatility_cap = volatility_cap or 0.006

    ticker = pd.read_excel("Green_ETF_Selection.xlsx", sheet_name="ETF_Universe", engine="openpyxl")
    tickers = list(ticker.Ticker)
    labels = list(ticker.Label)
    df = yfin.download(tickers, start=start_date, end=end_date)['Close']
    df.columns = labels
    df_cleaned = df.ffill().bfill()

    RA = RoboAdvisor(df_cleaned, rf='Green Bonds', benchmark='MSCI World SRI')
    logR = RA.getLogReturn(df_cleaned)

    sol = RA.optimizeWeights(
        logR=logR,
        strategy=strategy,
        mup=expected_return if expected_return is not None else 0.006,
        sigmap=volatility_cap if volatility_cap is not None else 0.006,
        printSol=False
    )

    mu = logR.mean()
    Cov = logR.cov()
    rf = RA.rf
    sharpe_ratio = round(((sol @ mu.T - rf) / (sol @ Cov @ sol.T) ** 0.5), 4)

    # ✅ Save chart visuals to static/charts
    RA.save_all_charts()

    # ✅ Save history to database
    if current_user.is_authenticated:
        from app.models import History
        new_entry = History(
            user_id=current_user.id,
            strategy=strategy,
            expected_return=expected_return,
            volatility_cap=volatility_cap,
            result_summary=f"Sharpe Ratio: {sharpe_ratio}"
        )
        db.session.add(new_entry)
        db.session.commit()

    # ✅ Compute log return histogram for extra chart
    total_returns = logR.sum(axis=1)
    hist, bin_edges = np.histogram(total_returns, bins=10)
    bin_centers = [round((bin_edges[i] + bin_edges[i+1]) / 2, 4) for i in range(len(bin_edges) - 1)]

    return render_template('results.html',
                           strategy=strategy,
                           expected_return=round((expected_return or 0.0) * 100, 2),
                           volatility_cap=round((volatility_cap or 0.0) * 100, 2),
                           sharpe_ratio=sharpe_ratio,
                           labels=list(df_cleaned.columns),
                           weights=[round(w, 4) for w in sol],
                           log_bins=bin_centers,
                           log_counts=hist.tolist())

