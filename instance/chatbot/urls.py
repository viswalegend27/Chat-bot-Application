from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('chat/', views.chat, name='chat'),
    path('upload/', views.upload_document, name='upload'),
    path('documents/', views.documents, name='documents'),
    path('delete_document/<int:doc_id>/', views.delete_document, name='delete_document'),
    path('clear_history/', views.clear_history, name='clear_history'),
    path('clear_documents/', views.clear_documents, name='clear_documents'),
    path('set_mode/', views.set_mode, name='set_mode'),
    path('history/', views.search_history, name='search_history'),
    path('debug/database/', views.debug_database, name='debug_database'),
]