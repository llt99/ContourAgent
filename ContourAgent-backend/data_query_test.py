import logging
import csv
from typing import Dict, Any
import os

# 日志配置
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# CSV文件配置
COAL_CSV_FILE = "coal.csv"

def load_coal_csv_data():
    """从coal.csv文件加载真实数据"""
    try:
        if not os.path.exists(COAL_CSV_FILE):
            logger.error(f"CSV文件 {COAL_CSV_FILE} 不存在")
            return []
        
        rows = []
        with open(COAL_CSV_FILE, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                # 转换数据类型
                processed_row = {
                    "well_name": row.get("well_name", ""),
                    "lon": float(row.get("lon", 0)),
                    "lat": float(row.get("lat", 0)),
                    "stratum_name": row.get("stratum_name", "龙潭"),
                    "stratum_thickness": float(row.get("stratum_thickness", 0)),
                    "lith_thickness": float(row.get("lith_thickness", 0)),
                    "ratio": float(row.get("ratio", 0))
                }
                rows.append(processed_row)
        
        logger.info(f"成功从 {COAL_CSV_FILE} 加载 {len(rows)} 条数据")
        return rows
    except Exception as e:
        logger.error(f"加载CSV文件失败: {e}")
        return []

def text_to_sql_query(query: str) -> Dict[str, Any]:
    """
    使用coal.csv数据的查询函数
    直接返回coal.csv中的所有数据，无需再检索
    """
    try:
        logger.info(f"🔧 使用coal.csv数据模式处理查询: {query}")
        
        # 直接从coal.csv加载数据
        raw_rows = load_coal_csv_data()
        
        # 转换数据格式为前端期望的格式
        rows = []
        for raw_row in raw_rows:
            row = {
                "well_name": raw_row.get("well_name", ""),
                "lon": raw_row.get("lon", 0),
                "lat": raw_row.get("lat", 0),
                "stratum_name": raw_row.get("stratum_name", "龙潭"),
                "stratum_thickness": raw_row.get("stratum_thickness", 0),
                "lith_thickness": raw_row.get("lith_thickness", 0),
                "ratio": raw_row.get("ratio", 0),
                "thickness": raw_row.get("stratum_thickness", 0)  # 兼容旧字段
            }
            rows.append(row)
        
        # 简单的SQL语句，表示数据来自coal.csv
        sql = "SELECT -- 数据来源: coal.csv\nSELECT well_name, lon, lat, stratum_name, stratum_thickness, lith_thickness, ratio FROM coal_data;"
        
        return {"sql": sql, "rows": rows, "rows_count": len(rows)}
            
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise RuntimeError(f"查询失败: {e}")

# ------------------------------
# 测试函数
# ------------------------------
if __name__ == "__main__":
    # 测试查询
    test_queries = [
        "绘制四川盆地龙潭组煤岩分布图",
        "Generate the coal rock distribution map of the Longtan Formation in the Sichuan Basin using Ordinary Kriging and a spherical model"
    ]
    
    print("=" * 60)
    print("测试coal.csv数据查询功能")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            result = text_to_sql_query(query)
            print(f"✅ 成功")
            print(f"   SQL: {result['sql'][:80]}...")
            print(f"   数据条数: {result['rows_count']}")
            if result['rows']:
                print(f"   示例数据: {result['rows'][0]}")
        except Exception as e:
            print(f"❌ 失败: {e}")
