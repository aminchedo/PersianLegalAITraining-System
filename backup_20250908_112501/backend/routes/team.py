from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from models.team_model import TeamMember
from config.database import get_db
from pydantic import BaseModel

router = APIRouter(prefix='/api/real/team', tags=['team'])

# Pydantic models for request/response
class TeamMemberCreate(BaseModel):
    name: str
    email: str
    role: str
    phone: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    experience_years: int = 0
    skills: List[str] = []
    permissions: List[str] = []
    projects: List[str] = []
    avatar: Optional[str] = None

class TeamMemberUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[int] = None
    skills: Optional[List[str]] = None
    permissions: Optional[List[str]] = None
    projects: Optional[List[str]] = None
    avatar: Optional[str] = None
    is_active: Optional[bool] = None

class TeamMemberResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    status: str
    phone: Optional[str]
    department: Optional[str]
    location: Optional[str]
    experience_years: int
    skills: List[str]
    permissions: List[str]
    projects: List[str]
    join_date: str
    last_active: Optional[str]
    is_active: bool
    avatar: Optional[str]
    total_tasks: int
    completed_tasks: int
    active_projects: int
    performance_score: float

    class Config:
        from_attributes = True

@router.get('/members', response_model=List[TeamMemberResponse])
async def get_team_members(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    department: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all team members with optional filtering"""
    query = db.query(TeamMember)
    
    if active_only:
        query = query.filter(TeamMember.is_active == True)
    
    if department:
        query = query.filter(TeamMember.department == department)
    
    members = query.offset(skip).limit(limit).all()
    return members

@router.get('/members/{member_id}', response_model=TeamMemberResponse)
async def get_team_member(member_id: int, db: Session = Depends(get_db)):
    """Get a specific team member by ID"""
    member = db.query(TeamMember).filter(TeamMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail='Team member not found')
    return member

@router.post('/members', response_model=TeamMemberResponse)
async def create_team_member(member_data: TeamMemberCreate, db: Session = Depends(get_db)):
    """Create a new team member"""
    # Check if email already exists
    existing_member = db.query(TeamMember).filter(TeamMember.email == member_data.email).first()
    if existing_member:
        raise HTTPException(status_code=400, detail='Email already exists')
    
    new_member = TeamMember(**member_data.dict())
    db.add(new_member)
    db.commit()
    db.refresh(new_member)
    return new_member

@router.put('/members/{member_id}', response_model=TeamMemberResponse)
async def update_team_member(
    member_id: int, 
    member_data: TeamMemberUpdate, 
    db: Session = Depends(get_db)
):
    """Update a team member"""
    member = db.query(TeamMember).filter(TeamMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail='Team member not found')
    
    # Update only provided fields
    update_data = member_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(member, field, value)
    
    db.commit()
    db.refresh(member)
    return member

@router.delete('/members/{member_id}')
async def delete_team_member(member_id: int, db: Session = Depends(get_db)):
    """Delete a team member (soft delete)"""
    member = db.query(TeamMember).filter(TeamMember.id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail='Team member not found')
    
    member.is_active = False
    db.commit()
    return {"message": "Team member deactivated successfully"}

@router.get('/stats')
async def get_team_stats(db: Session = Depends(get_db)):
    """Get team statistics"""
    total_members = db.query(TeamMember).count()
    active_members = db.query(TeamMember).filter(TeamMember.is_active == True).count()
    online_members = db.query(TeamMember).filter(
        TeamMember.is_active == True,
        TeamMember.status == 'online'
    ).count()
    
    # Department breakdown
    departments = db.query(TeamMember.department).filter(
        TeamMember.is_active == True,
        TeamMember.department.isnot(None)
    ).distinct().all()
    
    return {
        "total_members": total_members,
        "active_members": active_members,
        "online_members": online_members,
        "departments": [dept[0] for dept in departments]
    }