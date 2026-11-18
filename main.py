"""Flask entry point. Models are loaded only when a report needs them."""

import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

from config import Configuration
from src.v1.routes import biometric_v1


def create_app():
    app = Flask(__name__)
    app.config.from_object(Configuration)
    app.register_blueprint(biometric_v1, url_prefix='/biometric/v1')

    @app.get('/')
    @app.get('/health')
    def health():
        return {'service': 'ID/Passport Biometric Authentication', 'status': 'running'}

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8016')))
