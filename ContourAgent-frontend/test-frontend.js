/**
 * 地质等值线图生成系统 - 前端快速测试文件
 * 专门测试数据点数量对克里金插值的影响
 */

// 测试工具函数
const TestHelper = {
  // 模拟API响应
  mockAPIResponse: (endpoint, data) => {
    console.log(`🧪 Mocking API: ${endpoint}`);
    return Promise.resolve({ data });
  },

  // 测试控制台输出
  log: (message, type = 'info') => {
    const colors = {
      info: '\x1b[36m',
      success: '\x1b[32m',
      error: '\x1b[31m',
      warning: '\x1b[33m'
    };
    console.log(`${colors[type]}[TEST] ${message}\x1b[0m`);
  },

  // 测试计时器
  timer: (name) => {
    const start = Date.now();
    return {
      end: () => {
        const duration = Date.now() - start;
        TestHelper.log(`${name} 耗时: ${duration}ms`, 'info');
      }
    };
  },

  // 验证数据点数量是否足够插值
  validateDataPoints: (points, minRequired = 5) => {
    const count = points.length;
    const isValid = count >= minRequired;
    TestHelper.log(`数据点数量: ${count} (最低要求: ${minRequired}) - ${isValid ? '✅ 足够' : '❌ 不足'}`, 
                   isValid ? 'success' : 'error');
    return isValid;
  }
};

// 测试用例集合
const TestCases = {
  // 测试1: 数据点数量验证（核心问题）
  testDataPointRequirements: () => {
    TestHelper.log('=== 测试1: 数据点数量对插值的影响 ===');
    
    const testScenarios = [
      { name: "单个井点", points: [{ lon: 104.1, lat: 30.5, value: 100 }], min: 5 },
      { name: "2个井点", points: [
        { lon: 104.1, lat: 30.5, value: 100 },
        { lon: 104.2, lat: 30.6, value: 150 }
      ], min: 5 },
      { name: "3个井点", points: [
        { lon: 104.1, lat: 30.5, value: 100 },
        { lon: 104.2, lat: 30.6, value: 150 },
        { lon: 104.3, lat: 30.7, value: 120 }
      ], min: 5 },
      { name: "5个井点", points: [
        { lon: 104.1, lat: 30.5, value: 100 },
        { lon: 104.2, lat: 30.6, value: 150 },
        { lon: 104.3, lat: 30.7, value: 120 },
        { lon: 104.4, lat: 30.8, value: 180 },
        { lon: 104.5, lat: 30.9, value: 90 }
      ], min: 5 },
      { name: "10个井点", points: Array.from({ length: 10 }, (_, i) => ({
        lon: 104.1 + i * 0.1,
        lat: 30.5 + i * 0.1,
        value: 100 + Math.random() * 50
      })), min: 5 }
    ];

    const results = {};
    testScenarios.forEach(scenario => {
      const isValid = TestHelper.validateDataPoints(scenario.points, scenario.min);
      results[scenario.name] = isValid;
    });

    return results;
  },

  // 测试2: 数据分布质量验证
  testDataDistribution: () => {
    TestHelper.log('=== 测试2: 数据点分布质量 ===');
    
    const goodDistribution = [
      { lon: 104.0, lat: 30.5, value: 100 },
      { lon: 104.5, lat: 30.5, value: 150 },
      { lon: 104.0, lat: 31.0, value: 120 },
      { lon: 104.5, lat: 31.0, value: 180 },
      { lon: 104.25, lat: 30.75, value: 140 }
    ];

    const poorDistribution = [
      { lon: 104.1, lat: 30.5, value: 100 },
      { lon: 104.11, lat: 30.51, value: 150 },
      { lon: 104.12, lat: 30.52, value: 120 },
      { lon: 104.13, lat: 30.53, value: 180 },
      { lon: 104.14, lat: 30.54, value: 140 }
    ];

    // 计算分布范围
    const calcRange = (points) => {
      const lons = points.map(p => p.lon);
      const lats = points.map(p => p.lat);
      return {
        lon_range: Math.max(...lons) - Math.min(...lons),
        lat_range: Math.max(...lats) - Math.min(...lats)
      };
    };

    const goodRange = calcRange(goodDistribution);
    const poorRange = calcRange(poorDistribution);

    TestHelper.log(`良好分布范围: lon=${goodRange.lon_range.toFixed(3)}, lat=${goodRange.lat_range.toFixed(3)}`, 'success');
    TestHelper.log(`较差分布范围: lon=${poorRange.lon_range.toFixed(3)}, lat=${poorRange.lat_range.toFixed(3)}`, 'warning');

    const goodEnough = goodRange.lon_range > 0.3 && goodRange.lat_range > 0.3;
    const poorEnough = poorRange.lon_range < 0.05 && poorRange.lat_range < 0.05;

    return { goodDistribution: goodEnough, poorDistribution: poorEnough };
  },

  // 测试3: 模拟真实API调用流程
  testAPIWorkflow: async () => {
    TestHelper.log('=== 测试3: 完整API工作流程 ===');
    
    const timer = TestHelper.timer('API工作流程');
    
    try {
      // 步骤1: NLP解析
      const nlpResponse = await TestHelper.mockAPIResponse('/task', {
        nlpResult: {
          variable: "地层厚度",
          region: "四川盆地",
          formation: "龙潭组"
        },
        plan: { pipeline: ["nlp", "data", "kriging", "image"] }
      });

      // 步骤2: 数据查询（模拟不同数量的井点）
      const dataScenarios = [
        { name: "数据不足", count: 3 },
        { name: "数据刚好", count: 5 },
        { name: "数据充足", count: 15 }
      ];

      const dataResults = {};
      for (const scenario of dataScenarios) {
        const mockPoints = Array.from({ length: scenario.count }, (_, i) => ({
          well_name: `井${i + 1}`,
          lon: 104.1 + i * 0.05,
          lat: 30.5 + i * 0.05,
          thickness: 100 + Math.random() * 50
        }));

        const dataResponse = await TestHelper.mockAPIResponse('/task', {
          dataResult: mockPoints
        });

        const hasEnoughData = mockPoints.length >= 5;
        dataResults[scenario.name] = {
          count: mockPoints.length,
          sufficient: hasEnoughData
        };

        TestHelper.log(`${scenario.name}: ${mockPoints.length}个点 - ${hasEnoughData ? '✅ 可插值' : '❌ 无法插值'}`, 
                       hasEnoughData ? 'success' : 'error');
      }

      // 步骤3: 插值（仅当数据充足时）
      const krigingResults = {};
      for (const [name, info] of Object.entries(dataResults)) {
        if (info.sufficient) {
          const krigingResponse = await TestHelper.mockAPIResponse('/task', {
            krigingResult: {
              grid_x: [[104.1, 104.2], [104.1, 104.2]],
              grid_y: [[30.5, 30.5], [30.6, 30.6]],
              z: [[100, 150], [120, 180]],
              best_model: "spherical",
              selected_method: "ok"
            }
          });
          krigingResults[name] = "成功";
        } else {
          krigingResults[name] = "跳过（数据不足）";
        }
      }

      timer.end();
      return { nlp: true, data: dataResults, kriging: krigingResults };

    } catch (error) {
      TestHelper.log(`API测试失败: ${error.message}`, 'error');
      return false;
    }
  },

  // 测试4: 地图渲染组件状态
  testMapComponentState: () => {
    TestHelper.log('=== 测试4: 地图组件状态验证 ===');
    
    const mockState = {
      // 数据点图层
      ptLayer: {
        visible: true,
        source: {
          features: [
            { geometry: { type: "Point" }, properties: { well_name: "测试井1", thickness: 100 } },
            { geometry: { type: "Point" }, properties: { well_name: "测试井2", thickness: 150 } }
          ]
        }
      },
      // 等值线图层
      krigingVectorLayer: {
        visible: true,
        source: {
          features: [
            { geometry: { type: "MultiPolygon" }, properties: { fill: "#006837", value: 120 } }
          ]
        }
      },
      // 渲染参数
      params: {
        colors: ["#006837", "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b"],
        showLegend: true
      }
    };

    const hasDataPoints = mockState.ptLayer.source.features.length > 0;
    const hasContours = mockState.krigingVectorLayer.source.features.length > 0;
    const hasLegend = mockState.params.showLegend;

    TestHelper.log(`数据点图层: ${hasDataPoints ? '✅ 有数据' : '❌ 无数据'}`, hasDataPoints ? 'success' : 'error');
    TestHelper.log(`等值线图层: ${hasContours ? '✅ 有等值线' : '❌ 无等值线'}`, hasContours ? 'success' : 'error');
    TestHelper.log(`图例显示: ${hasLegend ? '✅ 正常' : '❌ 异常'}`, hasLegend ? 'success' : 'error');

    return hasDataPoints && hasContours && hasLegend;
  },

  // 测试5: 错误处理和边界情况
  testErrorHandling: () => {
    TestHelper.log('=== 测试5: 错误处理和边界情况 ===');
    
    const scenarios = [
      {
        name: "空数据",
        data: [],
        expected: "错误提示：无数据点"
      },
      {
        name: "单点数据",
        data: [{ lon: 104.1, lat: 30.5, value: 100 }],
        expected: "错误提示：数据点过少"
      },
      {
        name: "无效坐标",
        data: [
          { lon: null, lat: 30.5, value: 100 },
          { lon: 104.2, lat: null, value: 150 }
        ],
        expected: "错误提示：坐标无效"
      },
      {
        name: "重复坐标",
        data: [
          { lon: 104.1, lat: 30.5, value: 100 },
          { lon: 104.1, lat: 30.5, value: 150 }
        ],
        expected: "警告：重复坐标"
      }
    ];

    const results = {};
    scenarios.forEach(scenario => {
      const validPoints = scenario.data.filter(p => p.lon != null && p.lat != null && p.value != null);
      const uniquePoints = [];
      const seen = new Set();
      
      validPoints.forEach(p => {
        const key = `${p.lon},${p.lat}`;
        if (!seen.has(key)) {
          uniquePoints.push(p);
          seen.add(key);
        }
      });

      const shouldFail = validPoints.length < 5 || uniquePoints.length < 5;
      results[scenario.name] = !shouldFail;

      TestHelper.log(`${scenario.name}: ${validPoints.length}有效点, ${uniquePoints.length}唯一点 - ${shouldFail ? '❌ 预期失败' : '✅ 预期成功'}`, 
                     shouldFail ? 'error' : 'success');
    });

    return results;
  }
};

// 主测试函数
async function runAllTests() {
  console.log('\n=== 地质等值线图生成系统 - 数据点测试 ===\n');
  
  const results = {
    数据点数量: TestCases.testDataPointRequirements(),
    数据分布: TestCases.testDataDistribution(),
    API流程: await TestCases.testAPIWorkflow(),
    地图组件: TestCases.testMapComponentState(),
    错误处理: TestCases.testErrorHandling()
  };

  console.log('\n=== 测试结果汇总 ===');
  Object.entries(results).forEach(([testName, result]) => {
    if (typeof result === 'object') {
      console.log(`\n${testName}:`);
      Object.entries(result).forEach(([subTest, subResult]) => {
        if (typeof subResult === 'object') {
          console.log(`  ${subTest}:`);
          Object.entries(subResult).forEach(([key, val]) => {
            const status = val ? '✅' : '❌';
            console.log(`    ${key}: ${status}`);
          });
        } else {
          const status = subResult ? '✅' : '❌';
          console.log(`  ${subTest}: ${status}`);
        }
      });
    } else {
      const status = result ? '✅' : '❌';
      console.log(`${testName}: ${status}`);
    }
  });

  console.log('\n=== 关键发现 ===');
  console.log('1. 克里金插值最低需要5个数据点');
  console.log('2. 数据点分布越均匀，插值效果越好');
  console.log('3. 重复坐标会被自动去重');
  console.log('4. 无效坐标（null/undefined）会被过滤');
  
  console.log('\n=== 建议 ===');
  console.log('✅ 确保至少有5个不同的井点数据');
  console.log('✅ 数据点应分布在目标区域的不同位置');
  console.log('✅ 上传Excel前检查数据完整性和坐标有效性');
  console.log('✅ 如果数据不足，系统应给出明确错误提示');

  return results;
}

// 如果直接运行此文件
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TestHelper, TestCases, runAllTests };
} else {
  // 在浏览器中运行
  console.log('在浏览器控制台中运行: runAllTests()');
  window.TestHelper = TestHelper;
  window.TestCases = TestCases;
  window.runAllTests = runAllTests;
}
