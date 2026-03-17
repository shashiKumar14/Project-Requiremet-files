from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from fastapi import Depends

engine = create_engine("sqlite:///./db.sqlite3", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
def users(db=Depends(get_db)):
    # db.query(User).all()
    return []
