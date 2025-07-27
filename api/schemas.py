from pydantic import BaseModel
from typing import List

class DailyRecord(BaseModel):
    Date: str  # Format: 'YYYY-MM-DD'
    Product_ID: str
    Price: float
    Competitor_Pricing: float
    Discount: float
    Holiday_Promotion: int
    Units_Sold: float

class PredictionRequest(BaseModel):
    product_id: str
    data: List[DailyRecord]
