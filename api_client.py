"""
电力潮流计算API客户端工具
用于测试和调用后端服务
"""

import requests
import json
import numpy as np
from typing import Dict, List, Optional

class PowerFlowAPIClient:
    def __init__(self, base_url: str = "http://localhost:5001"):
        """
        初始化API客户端
        
        Args:
            base_url: 后端服务地址
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def save_parameters(self, base_voltage: float, base_power: float, convergence_precision: float) -> Dict:
        """保存计算参数"""
        data = {
            "baseVoltage": str(base_voltage),
            "basePower": str(base_power),
            "convergencePrecision": str(convergence_precision)
        }
        try:
            response = self.session.post(f"{self.base_url}/api/flow-compute/parameters", json=data)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_parameters(self) -> Dict:
        """获取保存的参数"""
        try:
            response = self.session.get(f"{self.base_url}/api/flow-compute/parameters")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_power_flow(self, bus_data: List[List], branch_data: List[List], 
                           base_voltage: float = 10.3, base_power: float = 10.38, 
                           convergence_precision: float = 1e-6) -> Dict:
        """
        执行潮流计算
        
        Args:
            bus_data: 节点数据 [[节点号, 负荷有功, 负荷无功], ...]
            branch_data: 支路数据 [[支路号, 首节点, 尾节点, 电阻, 电抗], ...]
            base_voltage: 基准电压 (kV)
            base_power: 基准功率 (MVA)
            convergence_precision: 收敛精度
        """
        data = {
            "busData": bus_data,
            "branchData": branch_data,
            "baseVoltage": base_voltage,
            "basePower": base_power,
            "convergencePrecision": convergence_precision
        }
        try:
            response = self.session.post(f"{self.base_url}/api/flow-compute/calculate", json=data)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_template(self) -> Dict:
        """获取示例数据模板"""
        try:
            response = self.session.get(f"{self.base_url}/api/flow-compute/template")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_results(self) -> Dict:
        """获取历史计算结果列表"""
        try:
            response = self.session.get(f"{self.base_url}/api/flow-compute/results")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_result_detail(self, filename: str) -> Dict:
        """获取具体计算结果详情"""
        try:
            response = self.session.get(f"{self.base_url}/api/flow-compute/results/{filename}")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def export_result(self, filename: str) -> Optional[bytes]:
        """导出计算结果"""
        try:
            response = self.session.get(f"{self.base_url}/api/flow-compute/export/{filename}")
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            print(f"导出失败: {e}")
            return None

def test_api():
    """测试API功能"""
    client = PowerFlowAPIClient()
    
    print("=== 电力潮流计算API测试 ===\n")
    
    # 1. 健康检查
    print("1. 健康检查...")
    health = client.health_check()
    print(f"结果: {health}\n")
    
    # 2. 获取模板数据
    print("2. 获取模板数据...")
    template = client.get_template()
    if template.get('status') == 'success':
        print("✓ 模板数据获取成功")
        bus_data = template['data']['busData']
        branch_data = template['data']['branchData']
        print(f"节点数量: {len(bus_data)}")
        print(f"支路数量: {len(branch_data)}")
    else:
        print(f"✗ 模板数据获取失败: {template.get('message')}")
        return
    print()
    
    # 3. 保存参数
    print("3. 保存参数...")
    params_result = client.save_parameters(10.3, 10.38, 1e-6)
    print(f"结果: {params_result}\n")
    
    # 4. 执行潮流计算
    print("4. 执行潮流计算...")
    calc_result = client.calculate_power_flow(
        bus_data=bus_data,
        branch_data=branch_data,
        base_voltage=10.3,
        base_power=10.38,
        convergence_precision=1e-6
    )
    
    if calc_result.get('status') == 'success':
        print("✓ 潮流计算成功")
        summary = calc_result.get('summary', {})
        print(f"迭代次数: {summary.get('iteration_count', 'N/A')}")
        print(f"是否收敛: {summary.get('converged', 'N/A')}")
        print(f"总损耗: {summary.get('total_active_loss_mw', 'N/A')} MW")
        print(f"网损率: {summary.get('loss_rate_percent', 'N/A')} %")
    else:
        print(f"✗ 潮流计算失败: {calc_result.get('message')}")
    print()
    
    # 5. 获取结果列表
    print("5. 获取结果列表...")
    results = client.get_results()
    if results.get('status') == 'success':
        print(f"✓ 找到 {len(results['data'])} 个历史结果")
        for result in results['data'][:3]:  # 显示前3个
            print(f"  - {result['filename']} ({result['timestamp']})")
    else:
        print(f"✗ 获取结果列表失败: {results.get('message')}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_api()
