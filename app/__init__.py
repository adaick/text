from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
import os

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
migrate = Migrate()

# Optional: Secure admin model access
class SecureModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated and current_user.email == "admin@nava.com"  # change to your admin email

    def inaccessible_callback(self, name, **kwargs):
        from flask import redirect, url_for
        return redirect(url_for('routes.home'))

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with env-secured key in production
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    # Import and register routes
    from app import routes, models
    app.register_blueprint(routes.bp)

    # ✅ Flask-Admin setup
    admin = Admin(app, name='NAVA Admin', template_mode='bootstrap4')
    admin.add_view(SecureModelView(models.User, db.session))
    admin.add_view(SecureModelView(models.History, db.session))

    return app
