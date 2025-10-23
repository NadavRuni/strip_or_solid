from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict
import json

@dataclass
class White_Score:
    white_score_test_1: float = 0.0
    white_score_test_2: float = 0.0
    white_score_test_3: float = 0.0
    white_score_test_4: float = 0.0
    white_score_test_5: float = 0.0

@dataclass
class Black_Score:
    black_score_test_1: float = 0.0
    black_score_test_2: float = 0.0
    black_score_test_3: float = 0.0
    black_score_test_4: float = 0.0
    black_score_test_5: float = 0.0

@dataclass
class Solid_Score:
    solid_score_test_1: float = 0.0
    solid_score_test_2: float = 0.0
    solid_score_test_3: float = 0.0
    solid_score_test_4: float = 0.0
    solid_score_test_5: float = 0.0

@dataclass
class Striped_Score:
    striped_score_test_1: float = 0.0
    striped_score_test_2: float = 0.0
    striped_score_test_3: float = 0.0
    striped_score_test_4: float = 0.0
    striped_score_test_5: float = 0.0

@dataclass
class Color_Score:
    white_score : White_Score =field(default_factory=White_Score)
    black_score : Black_Score = field(default_factory=Black_Score)
    solid_score : Solid_Score = field(default_factory=Solid_Score)
    striped_score : Striped_Score = field(default_factory=Striped_Score)

@dataclass
class Ball_Color:
    WHITE = "white"
    BLACK = "black"
    SOLID = "solid"
    STRIPED = "striped"
    UNDEFINED = "undefined"

@dataclass
class Ball:
    center: Tuple[int, int]
    radius: int
    color_score: Color_Score = field(default_factory=Color_Score)
    final_color: Ball_Color = Ball_Color.UNDEFINED
    single_ball_path: str = ""

@dataclass
class Pockets_img_paths:
    top_left_path: str =""
    top_right_path: str =""
    bottom_left_path: str =""
    bottom_right_path: str =""
    top_middle_path: str =""
    buttom_middle_path: str =""

@dataclass
class Pocket_Location_On_Table:
    top_left: str ="TL"
    top_right: str ="TR"
    bottom_left: str ="BL"
    bottom_right: str ="BR"
    top_middle: str ="TM"
    buttom_middle: str ="BM"
    unknown: str ="UNKNOWN"


@dataclass
class Pocket :
    pocket_id : int
    pocket_center :Tuple[int,int]
    pocker_radius : int
    pocket_img_path : str
    pocket_img_cordinates_on_table : Tuple[int,int] 
    pocket_loacation_on_table : Pocket_Location_On_Table = Pocket_Location_On_Table.unknown



@dataclass
class table_pockets:
    pockets_img_paths: Pockets_img_paths = field(default_factory=Pockets_img_paths)
    pocket_list: List[Pocket] = field(default_factory=list)




@dataclass
class AnalyzerResult:
    black: Optional[Ball] = None
    white: Optional[Ball] = None
    Pockets: table_pockets = field(default_factory=table_pockets)
    balls: List[Ball] = field(default_factory=list)


    
@dataclass
class Rectangle:
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]



@dataclass
class Origin:
    x: int
    y: int

@dataclass
class PhotoData:
    cut_name: str
    origin: Origin
    rectangle: Rectangle
    balls: List[Ball]

    def to_dict(self) -> Dict:
        """המרת האובייקט למילון רגיל לצורך שמירה ל־JSON"""
        return asdict(self)

    def save_json(self, path: str):
        """שמירת האובייקט לקובץ JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_json(path: str) -> "PhotoData":
        """טעינת קובץ JSON לאובייקט מסוג PhotoData"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        balls = [Ball(tuple(ball["center"]), ball["radius"]) for ball in data.get("balls", [])]
        rect = data["rectangle"]
        rectangle = Rectangle(
            tuple(rect["top_left"]),
            tuple(rect["top_right"]),
            tuple(rect["bottom_left"]),
            tuple(rect["bottom_right"]),
        )
        origin = Origin(data["origin"]["x"], data["origin"]["y"])

        return PhotoData(
            cut_name=data["cut_name"],
            origin=origin,
            rectangle=rectangle,
            balls=balls
        )
