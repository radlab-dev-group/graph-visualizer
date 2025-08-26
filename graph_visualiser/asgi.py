from asgiref.wsgi import WsgiToAsgi
from app import server

asgi_app = WsgiToAsgi(server)
