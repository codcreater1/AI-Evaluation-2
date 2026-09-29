from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import SystemIn, SystemOut
from app.db.models import System
from app.db.session import get_db
from app.evaluators import list_evaluators

router = APIRouter(tags=["systems"])


@router.get("/systems", response_model=list[SystemOut])
def get_systems(db: Session = Depends(get_db)):
    return db.scalars(select(System).order_by(System.name)).all()


@router.post("/systems", response_model=SystemOut, status_code=201)
def register_system(body: SystemIn, db: Session = Depends(get_db)):
    obj = db.get(System, body.name) or System(name=body.name)
    obj.description, obj.endpoint_url, obj.config = body.description, body.endpoint_url, body.config
    db.add(obj)
    db.commit()
    return obj


@router.get("/evaluators")
def get_evaluators():
    return {"evaluators": list_evaluators()}
