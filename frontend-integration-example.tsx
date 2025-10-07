/**
 * 前端集成示例 - FlowCompute组件增强版
 * 展示如何将React前端与后端API进行集成
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Item,
  ItemContent,
  ItemTitle,
  ItemActions,
  ItemMedia,
} from '@/components/ui/item';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import {
  WavesIcon,
  Settings,
  Eye,
  SaveIcon,
  Calculator,
  FileDown,
  Bug,
  Loader2,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import { ButtonGroup } from '../ui/button-group';

// 定义表单验证模式
const flowComputeSchema = z.object({
  baseVoltage: z
    .string()
    .min(1, '请输入基准电压')
    .refine(val => !isNaN(Number(val)) && Number(val) > 0, {
      message: '基准电压必须为正数',
    }),
  basePower: z
    .string()
    .min(1, '请输入基准功率')
    .refine(val => !isNaN(Number(val)) && Number(val) > 0, {
      message: '基准功率必须为正数',
    }),
  convergencePrecision: z
    .string()
    .min(1, '请输入收敛精度')
    .refine(val => !isNaN(Number(val)) && Number(val) > 0, {
      message: '收敛精度必须为正数',
    }),
});

type FlowComputeFormData = z.infer<typeof flowComputeSchema>;

// API客户端类
class PowerFlowAPI {
  private baseURL = 'http://localhost:5000';
  
  async healthCheck() {
    const response = await fetch(`${this.baseURL}/api/health`);
    return response.json();
  }
  
  async saveParameters(data: FlowComputeFormData) {
    const response = await fetch(`${this.baseURL}/api/flow-compute/parameters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return response.json();
  }
  
  async getParameters() {
    const response = await fetch(`${this.baseURL}/api/flow-compute/parameters`);
    return response.json();
  }
  
  async getTemplate() {
    const response = await fetch(`${this.baseURL}/api/flow-compute/template`);
    return response.json();
  }
  
  async calculatePowerFlow(params: {
    baseVoltage: number;
    basePower: number;
    convergencePrecision: number;
    busData: number[][];
    branchData: number[][];
  }) {
    const response = await fetch(`${this.baseURL}/api/flow-compute/calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return response.json();
  }
  
  async getResults() {
    const response = await fetch(`${this.baseURL}/api/flow-compute/results`);
    return response.json();
  }
  
  async exportResult(filename: string) {
    const response = await fetch(`${this.baseURL}/api/flow-compute/export/${filename}`);
    if (response.ok) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
  }
}

export default function FlowComputeEnhanced() {
  const [api] = useState(new PowerFlowAPI());
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [calculationResult, setCalculationResult] = useState<any>(null);
  const [templateData, setTemplateData] = useState<any>(null);
  const [results, setResults] = useState<any[]>([]);

  const form = useForm<FlowComputeFormData>({
    resolver: zodResolver(flowComputeSchema),
    defaultValues: {
      baseVoltage: '10.3',
      basePower: '10.38',
      convergencePrecision: '1e-6',
    },
  });

  // 检查API状态
  useEffect(() => {
    const checkAPIStatus = async () => {
      try {
        const result = await api.healthCheck();
        setApiStatus(result.status === 'success' ? 'online' : 'offline');
      } catch (error) {
        setApiStatus('offline');
      }
    };
    
    checkAPIStatus();
    const interval = setInterval(checkAPIStatus, 30000); // 每30秒检查一次
    
    return () => clearInterval(interval);
  }, [api]);

  // 加载模板数据
  useEffect(() => {
    const loadTemplate = async () => {
      try {
        const result = await api.getTemplate();
        if (result.status === 'success') {
          setTemplateData(result.data);
        }
      } catch (error) {
        console.error('加载模板数据失败:', error);
      }
    };
    
    if (apiStatus === 'online') {
      loadTemplate();
    }
  }, [api, apiStatus]);

  // 加载历史结果
  useEffect(() => {
    const loadResults = async () => {
      try {
        const result = await api.getResults();
        if (result.status === 'success') {
          setResults(result.data);
        }
      } catch (error) {
        console.error('加载结果列表失败:', error);
      }
    };
    
    if (apiStatus === 'online') {
      loadResults();
    }
  }, [api, apiStatus]);

  // 保存参数
  const onSubmit = async (data: FlowComputeFormData) => {
    if (apiStatus !== 'online') {
      alert('后端服务未连接，请检查服务状态');
      return;
    }

    setLoading(true);
    try {
      const result = await api.saveParameters(data);
      if (result.status === 'success') {
        alert('参数保存成功！');
      } else {
        alert(`参数保存失败: ${result.message}`);
      }
    } catch (error) {
      alert(`保存失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 启动计算
  const handleCalculate = async () => {
    if (apiStatus !== 'online') {
      alert('后端服务未连接，请检查服务状态');
      return;
    }

    if (!templateData) {
      alert('模板数据未加载，请稍后重试');
      return;
    }

    const formData = form.getValues();
    setLoading(true);
    
    try {
      const result = await api.calculatePowerFlow({
        baseVoltage: parseFloat(formData.baseVoltage),
        basePower: parseFloat(formData.basePower),
        convergencePrecision: parseFloat(formData.convergencePrecision),
        busData: templateData.busData,
        branchData: templateData.branchData,
      });
      
      if (result.status === 'success') {
        setCalculationResult(result);
        // 重新加载结果列表
        const resultsResponse = await api.getResults();
        if (resultsResponse.status === 'success') {
          setResults(resultsResponse.data);
        }
        alert('潮流计算完成！');
      } else {
        alert(`计算失败: ${result.message}`);
      }
    } catch (error) {
      alert(`计算失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 查看结果
  const handleViewResults = () => {
    if (results.length === 0) {
      alert('暂无计算结果');
      return;
    }
    // 这里可以打开结果查看模态框或跳转到结果页面
    console.log('查看结果:', results[0]);
  };

  // 导出结果
  const handleExportResults = async () => {
    if (results.length === 0) {
      alert('暂无计算结果');
      return;
    }
    
    try {
      await api.exportResult(results[0].filename);
      alert('结果导出成功！');
    } catch (error) {
      alert(`导出失败: ${error}`);
    }
  };

  return (
    <section className="mx-auto px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
      >
        <CardHeader className="mb-4">
          <CardTitle className="text-xl md:text-2xl flex items-center gap-2 text-white/70">
            <WavesIcon />
            潮流计算
            {/* API状态指示器 */}
            <div className="ml-auto flex items-center gap-2">
              {apiStatus === 'checking' && <Loader2 className="h-4 w-4 animate-spin" />}
              {apiStatus === 'online' && <CheckCircle className="h-4 w-4 text-green-500" />}
              {apiStatus === 'offline' && <AlertCircle className="h-4 w-4 text-red-500" />}
              <span className="text-xs">
                {apiStatus === 'checking' && '检查中'}
                {apiStatus === 'online' && '在线'}
                {apiStatus === 'offline' && '离线'}
              </span>
            </div>
          </CardTitle>
          <CardDescription className="text-sm md:text-base">
            基于前推回代法的配网潮流计算，获取线路节点电压、相位以及支路功率损耗分布
          </CardDescription>
        </CardHeader>
        
        <Card className="bg-background/10">
          <CardContent>
            <div className="flex flex-col gap-4">
              <Form {...form}>
                <form
                  onSubmit={form.handleSubmit(onSubmit)}
                  className="h-full flex-1"
                >
                  <div className="flex gap-2 mb-4">
                    <ItemMedia variant="icon">
                      <Settings className="h-4 w-4" />
                    </ItemMedia>
                    <ItemTitle className="text-white/70">参数配置</ItemTitle>
                  </div>

                  <Item variant="outline" className="bg-white/5">
                    <ItemContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <FormField
                          control={form.control}
                          name="baseVoltage"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-sm">
                                基准电压 (kV)
                              </FormLabel>
                              <FormControl>
                                <Input
                                  {...field}
                                  placeholder="请输入基准电压"
                                  className="mt-2 bg-black/40 border-white/20 text-white placeholder:text-white/50"
                                  disabled={loading}
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                        <FormField
                          control={form.control}
                          name="basePower"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-sm">
                                基准功率 (MVA)
                              </FormLabel>
                              <FormControl>
                                <Input
                                  {...field}
                                  placeholder="请输入基准功率"
                                  className="mt-2 bg-black/40 border-white/20 text-white placeholder:text-white/50"
                                  disabled={loading}
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                        <FormField
                          control={form.control}
                          name="convergencePrecision"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-sm">
                                收敛精度
                              </FormLabel>
                              <FormControl>
                                <Input
                                  {...field}
                                  placeholder="请输入收敛精度"
                                  className="mt-2 bg-black/40 border-white/20 text-white placeholder:text-white/50"
                                  disabled={loading}
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>
                    </ItemContent>

                    <ItemActions className="self-end">
                      <ButtonGroup>
                        <Button 
                          variant="default" 
                          type="submit"
                          disabled={loading || apiStatus !== 'online'}
                        >
                          {loading ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <SaveIcon className="h-4 w-4" />
                          )}
                          保存参数
                        </Button>
                        <Button
                          variant="outline"
                          className="flex items-center gap-2"
                          disabled={loading || !templateData}
                        >
                          <Eye className="h-4 w-4" />
                          查看拓扑
                        </Button>
                      </ButtonGroup>
                    </ItemActions>
                  </Item>
                </form>
              </Form>

              <div className="mt-6 flex gap-4 justify-end">
                <Button 
                  variant="default"
                  onClick={handleCalculate}
                  disabled={loading || apiStatus !== 'online' || !templateData}
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Calculator className="h-4 w-4" />
                  )}
                  启动计算
                </Button>
                <Button 
                  variant="outline"
                  onClick={handleViewResults}
                  disabled={loading || results.length === 0}
                >
                  <Eye className="h-4 w-4" />
                  查看结果 ({results.length})
                </Button>
                <Button 
                  variant="outline"
                  onClick={handleExportResults}
                  disabled={loading || results.length === 0}
                >
                  <FileDown className="h-4 w-4" />
                  导出结果
                </Button>
              </div>

              {/* 计算结果预览 */}
              {calculationResult && (
                <div className="mt-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <h4 className="text-green-400 font-semibold mb-2">最新计算结果</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-white/70">迭代次数:</span>
                      <span className="ml-2 text-white">
                        {calculationResult.summary?.iteration_count || 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-white/70">是否收敛:</span>
                      <span className="ml-2 text-white">
                        {calculationResult.summary?.converged ? '是' : '否'}
                      </span>
                    </div>
                    <div>
                      <span className="text-white/70">总损耗:</span>
                      <span className="ml-2 text-white">
                        {calculationResult.summary?.total_active_loss_mw?.toFixed(4) || 'N/A'} MW
                      </span>
                    </div>
                    <div>
                      <span className="text-white/70">网损率:</span>
                      <span className="ml-2 text-white">
                        {calculationResult.summary?.loss_rate_percent?.toFixed(2) || 'N/A'} %
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </section>
  );
}
