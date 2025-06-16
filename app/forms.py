from flask_wtf import FlaskForm
from wtforms import StringField

class SearchForm(FlaskForm):
    query = StringField('query')
