<template xmlns="http://www.w3.org/1999/html">
<!--  <HeaderBar />-->
  <div class="page">
    <!-- 左侧问答区 -->
    <div class="qa-card">
      <!-- 左上按钮和标题 -->
      <div class="header-with-history">
        <button @click="showHistory = !showHistory" class="history-btn">
          📜
        </button>
        <h2 class="title">GeoContour AI</h2>
      </div>

      <!-- 聊天区 -->
      <div class="chat-box" ref="chatBox">
        <transition name="slide-right">
          <div v-if="showHistory"
               class="history-panel absolute top-0 left-0 bottom-0 w-72 bg-white shadow-lg p-4 overflow-y-auto"
               style="border-radius: 0 12px 12px 0;">
            <h4 class="font-bold mb-4">历史任务</h4>
            <!--                <h4 class="font-bold mb-4">History</h4>-->
            <ul>
              <li v-for="(item, index) in historyList" :key="index"
                  class="history-item cursor-pointer hover:bg-gray-100 p-2 rounded"
                  @click="applyHistory(item)">
                <div class="text-sm text-gray-600">{{ formatTimestamp(item.timestamp) }}</div>
                <div class="text-gray-800">{{ item.text }}</div>
              </li>
            </ul>
          </div>
        </transition>

        <div class="messages flex-1 ml-72 p-2 overflow-y-auto" style="max-height: 100%;">
          <div v-for="(msg, index) in messages" :key="index" :class="['msg', msg.role]">
            <span class="role-label">{{ msg.role === 'user' ? '🧑' : '🤖' }}</span>
            <div v-if="msg.type === 'image'" class="chat-image">
              <img
                  :src="'data:image/png;base64,' + msg.src"
                  style="max-width:200px; max-height:150px; border-radius:4px; cursor:pointer;"
                  @click="previewSrc = 'data:image/png;base64,' + msg.src"
              />
            </div>
            <div v-else class="text">{{ msg.text }}</div>
          </div>
        </div>

        <div v-if="previewSrc" class="preview-overlay" @click="previewSrc = null">
          <img :src="previewSrc" class="preview-img"/>
        </div>
      </div>

      <div class="input-area">
        <!-- 文件上传（支持数据和沉积相规则） -->
        <input
            type="file"
            ref="fileInput"
            accept=".xlsx, .xls"
            style="display: none"
            @change="handleFileUpload"
        />

        <button class="circle-btn" @click="$refs.fileInput.click()" title="上传Excel文件（数据或沉积相规则）">
          +
        </button>

        <input
            type="text"
            v-model="userInput"
            placeholder='Please enter your mapping request, for example: "Generate a thickness map of the Longtan Formation in the Sichuan Basin."'
            @keyup.enter="sendMessage"
        />
        <button @click="sendMessage">Send</button>
      </div>
    </div>
    <!--    placeholder="Please enter your mapping request, for example: “Generate a thickness map of the Longtan Formation in the Sichuan Basin.”"-->

    <!--            placeholder="请输入编图需求，例如“绘制四川盆地龙潭组地层厚度图”"-->

    <!-- 右侧地图区 -->
    <div class="map-card" ref="mapContainer">
      <button class="map-download-btn" @click="downloadMap">⬇ ADD MAP</button>
      <!-- ✅ 垂直离散图例 -->
      <div class="legend-card-vertical" v-if="legendData">
        <div class="legend-title">Legend</div>
        <div class="legend-content">
          <div class="legend-colors">
            <div v-for="(item, index) in legendData.items" :key="index"
                 class="legend-color-box"
                 :style="{ backgroundColor: item.color }">
            </div>
          </div>
          <div class="legend-labels-vertical">
            <span v-for="(item, index) in legendData.items" :key="index">
              {{ item.label }}
            </span>
          </div>
        </div>
      </div>
    </div>

  </div>
<!--  <FootBar />-->
</template>

<script>
import HeaderBar from "../components/HeaderBar.vue";
import FootBar from "../components/FootBar.vue";
import * as XLSX from 'xlsx'
import 'ol/ol.css'
import { Map, View } from 'ol'
import { Tile as TileLayer, Vector as VectorLayer, Image as ImageLayer } from 'ol/layer'
import { XYZ, Vector as VectorSource, ImageCanvas, ImageStatic } from 'ol/source'
import GeoJSON from 'ol/format/GeoJSON'
import { Point, MultiPolygon } from 'ol/geom'
import Feature from 'ol/Feature'
import axios from 'axios'
import chroma from "chroma-js";
import { Style, Circle as CircleStyle, Fill, Stroke, Text } from 'ol/style';

const API_BASE = 'http://10.242.48.50:8000';

export default {
  components: { HeaderBar, FootBar },
  data() {
    return {
      userInput: '',
      messages: [
        // { role: 'assistant', text: '您好，我是地质等值线图智能助手，请描述您的编图需求，例如“绘制四川盆地龙潭组地层厚度图”' }
        { role: 'assistant', text: 'Hello, I am the Geological Contour Map Intelligent Assistant. Please describe your mapping request, for example: “Draw a thickness map of the Longtan Formation in the Sichuan Basin.” ' }
      ],
      previewSrc: null,
      nlpResult: null,

      map: null,
      ptLayer: null,
      krigingVectorLayer: null,
      krigingCanvasLayer: null,
      basinLayer: null,
      krigingVectorSource: new VectorSource(),
      dataset: { type: 'FeatureCollection', features: [] },
      format: new GeoJSON(),
      params: { colors: [
          "#006837", "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf",
          "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"
        ]
      },

      showHistory: false,
      historyList: [],
      currentImage: null,
      currentGeoJSON: null,
      vectorLayer: null,

      geojsonResult: { map: { features: [], contour_levels: [] } },
      showLegend: false,
      contourLevels: [],
      colorScale: null,

      showVariogram: true,
      geojson: null,
      isZoomed: false,

      latestMapImage: null,
    }
  },
  computed: {
    legendData() {
      if (!this.showLegend) return null;
      const geo = this.geojsonResult;
      if (!geo || geo.type !== "FeatureCollection" || !geo.properties) return null;

      const { colormap, colors, contour_levels } = geo.properties;
      if (!colors || !contour_levels || colors.length === 0 || contour_levels.length === 0) return null;

      const reversedColors = [...colors].reverse();
      const reversedLevels = [...contour_levels].reverse();

      return {
        colormap: colormap || "Color Map",
        items: reversedColors.map((color, index) => ({
          color,
          label: reversedLevels[index] ? reversedLevels[index].toFixed(2) : ''
        })),
      };
    }
  },
  mounted() {
    // 离线地图瓦片层 - 使用正确的路径格式
    const tileLayer = new TileLayer({
      source: new XYZ({ 
        url: '/tiles/{z}/{x}/{y}.png',
        minZoom: 0,
        maxZoom: 12  // 根据public/tiles目录结构，最高支持到12级
      }),
      zIndex: 0
    });

    this.ptLayer = new VectorLayer({
      source: new VectorSource(),
      zIndex: 10,
      style: new Style({ image: new CircleStyle({ radius: 5, fill: new Fill({ color: '#0013f8' }) }) })
    });

    this.krigingVectorLayer = new VectorLayer({
      source: this.krigingVectorSource,
      zIndex: 4,
      style: feature => {
        const fillColor = feature.get('fill') || 'rgba(204, 204, 204, 0.5)'; // Fallback to a semi-transparent gray
        return new Style({
          // For contour maps, we usually only want a fill, not a stroke.
          // The stroke can interfere with the visualization of the colored bands.
          fill: new Fill({ color: fillColor })
        });
      }
    });


    this.krigingCanvasLayer = new ImageLayer({ zIndex: 5 });

    // 盆地边界 - 处理文件可能不存在的情况
    const fillColors = ['rgba(255,0,68,0.3)','rgba(19,246,83,0.3)','rgba(0,233,255,0.3)','rgba(229,246,6,0.3)','rgba(234,24,234,0.3)'];
    this.basinLayer = new VectorLayer({
      source: new VectorSource({ 
        url: '/scBasin.geojson', 
        format: new GeoJSON(),
        // 添加加载错误处理
        strategy: 'all'
      }),
      style: f => {
        const index = f.ol_uid ? f.ol_uid % fillColors.length : 0;
        return new Style({ fill: new Fill({ color: fillColors[index] }), stroke: new Stroke({ color: '#000', width: 0.2 }) });
      },
      zIndex: 2
    });
    
    // 监听盆地图层加载错误
    this.basinLayer.getSource().on('featuresloaderror', () => {
      console.warn('⚠️ 盆地边界文件(scBasin.geojson)不存在，已跳过');
    });

    try {
      this.map = new Map({
        target: this.$refs.mapContainer,
        layers: [ tileLayer, this.basinLayer, this.ptLayer, this.krigingVectorLayer, this.krigingCanvasLayer ],
        view: new View({ center: [106.5,30.35], projection:'EPSG:4326', zoom:7.5 })
      });
      
      // 监听地图加载事件，用于调试
      this.map.on('loaderror', (event) => {
        console.error('地图加载错误:', event);
      });
      
      console.log('✅ 地图初始化成功');
    } catch (error) {
      console.error('❌ 地图初始化失败:', error);
    }

    // 读取历史记录
    this.fetchHistory()

  },
  methods: {
    async sendMessage() {
      const content = this.userInput.trim();
      if (!content) return;
      this.messages.push({ role: 'user', text: content });
      this.userInput = '';

      try {
        const res = await axios.post(`${API_BASE}/task`, {
          text: content,
          feedback: null
        }, { timeout: 180000 });
        const data = res.data;

        console.log("== raw data ==", data);
        console.log("== geojsonResult ==", data.geojsonResult);


        // ================= 点数据渲染 =================
        let pointsData = [];
        if (Array.isArray(data.dataResult)) {
          pointsData = data.dataResult;
        } else if (data.dataResult?.rows?.length) {
          pointsData = data.dataResult.rows;
        }

        if (pointsData.length > 0) {
          this.ptLayer.getSource().clear();
          pointsData.forEach(p => {
            const lon = parseFloat(p.lon ?? p.geo_X);
            const lat = parseFloat(p.lat ?? p.geo_Y);
            const value = parseFloat(p.thickness ?? p.value ?? 0);

            console.log('井数据:', p, '经纬度:', lon, lat, '厚度:', value);

            if (!isNaN(lon) && !isNaN(lat) && !isNaN(value)) {
              const feature = new Feature({
                geometry: new Point([lon, lat]),
                value,
                name: p.name || p.well_name || '未知'
              });

              // 给点加样式（圆点 + 名称标注）

              feature.setStyle(new Style({
                image: new CircleStyle({
                  radius: 5,
                  fill: new Fill({ color: 'red' }),
                  stroke: new Stroke({ color: 'white', width: 1 })
                }),
                text: new Text({
                  text: feature.get('name') || '',
                  font: '12px Calibri,sans-serif',
                  offsetY: -12,
                  fill: new Fill({ color: '#000' }),
                  stroke: new Stroke({ color: '#fff', width: 2 })
                })
              }));


              this.ptLayer.getSource().addFeature(feature);
            }
          });
          console.log('共渲染点数:', this.ptLayer.getSource().getFeatures().length);
        }

        // ================= 等值线 GeoJSON 渲染 =================
        this.krigingVectorLayer.setVisible(true); // 每次请求时重置为可见

        let geojson = data.geojsonResult;

        // 1. 必须要求是 FeatureCollection，不进入 fallback
        if (!geojson || geojson.type !== "FeatureCollection") {
          console.error("❌ 返回的 geojsonResult 不是 FeatureCollection:", geojson);
          return;
        }

        // 2. 如果是字符串 → JSON.parse
        if (typeof geojson === "string") {
          try {
            geojson = JSON.parse(geojson);
          } catch (e) {
            console.error("❌ GeoJSON 字符串无法解析:", e, geojson);
            return;
          }
        }

        // 3. 保存原始数据（legend 要用）
        this.geojsonResult = geojson;

        // 4. OpenLayers 读取 Feature（投影转换）
        let featuresOL = [];
        try {
          featuresOL = new GeoJSON().readFeatures(geojson, {
            // 后端是 WGS-84（经纬度）
            dataProjection: "EPSG:4326",
            featureProjection: "EPSG:4326"
          });
        } catch (e) {
          console.error("❌ OpenLayers 无法读取 GeoJSON:", e, geojson);
          return;
        }

        // 5. 给 feature 设置 fill（必须，否则全灰）
        featuresOL.forEach(f => {
          const p = f.getProperties();
          const color =
              p.fill ||
              p.color ||
              (geojson.properties?.colors?.[0]) ||
              "#cccccc";
          f.set("fill", color);
        });

        // 6. 渲染到图层
        this.krigingVectorSource.clear();
        this.krigingVectorSource.addFeatures(featuresOL);

        // 7. 打开图例
        this.showLegend = true;

        console.log("🎉 GeoJSON 加载成功：", featuresOL.length, "个等值面");

        // ================= 图像渲染 =================
        if (data.imageResult) {
          let imageBase64;
          if (typeof data.imageResult === 'string') {
            imageBase64 = data.imageResult;
          } else {
            const imgKey = Object.keys(data.imageResult)[0];
            if (imgKey && data.imageResult[imgKey]) {
              imageBase64 = data.imageResult[imgKey];
            }
          }

          if (imageBase64) {
            this.latestMapImage = imageBase64;
            this.messages.push({
              role: 'assistant',
              type: 'image',
              src: imageBase64
            });
          }
        }

        // this.messages.push({ role: 'assistant', text: '✅ 等值线图已生成，请查看右侧地图' });
        this.messages.push({ role: 'assistant', text: '✅ The contour map has been generated. Please check the map on the right.' });

        // 滚动聊天框到底部
        this.$nextTick(() => {
          this.$refs.chatBox.scrollTop = this.$refs.chatBox.scrollHeight;
        });

      } catch (err) {
        console.error('axios error:', err);
        this.messages.push({ role: 'assistant', text: '❌ 服务请求失败: ' + (err.message || err) });
      }
    },

    async handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();

      reader.onload = async (e) => {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const json = XLSX.utils.sheet_to_json(worksheet);

          console.log('Excel data:', json);

        // 检测文件类型：数据文件还是沉积相规则文件
        // 简单启发式：检查第一行对象的键名是否包含规则相关关键词
        const firstRow = json[0] || {};
          const keys = Object.keys(firstRow);
          const isFaciesRules = keys.some(key =>
          key.toLowerCase().includes('规则') ||
          key.toLowerCase().includes('rule') ||
          key.toLowerCase().includes('facies') ||
          key.toLowerCase().includes('分类') ||
          key.toLowerCase().includes('标准') ||
          key.toLowerCase().includes('相') || // 中文沉积相相关
          key.toLowerCase().includes('亚相') ||
          key.toLowerCase().includes('含量') ||
          key.toLowerCase().includes('岩性')
          );

          if (isFaciesRules) {
          // 沉积相规则文件
            this.messages.push({ role: 'assistant', text: '📋 Facies classification rules uploaded, processing...' });

            try {
              const res = await axios.post(`${API_BASE}/upload-facies-rules`, {
              rules: json
            }, { timeout: 180000 });

              const responseData = res.data;
              console.log('Backend response for facies rules upload:', responseData);

              if (responseData.success) {
              this.messages.push({ role: 'assistant', text: '✅ Facies classification rules processed successfully. You can now generate maps using these rules.' });
              // 可以存储规则供后续使用
              if (responseData.rules) {
                // 存储规则到组件状态（如果需要）
              }
              } else {
                this.messages.push({ role: 'assistant', text: '❌ Backend failed to process facies rules: ' + (responseData.message || 'Unknown error') });
              }
            } catch (err) {
            console.error('axios error during facies rules upload:', err);
            this.messages.push({ role: 'assistant', text: '❌ Failed to send facies rules to backend: ' + (err.message || err) });
            }

          } else {
            this.messages.push({ role: 'assistant', text: '📊 Excel data file uploaded, processing data...' });

            try {
              const res = await axios.post(`${API_BASE}/upload-excel`, {
              data: json
            }, { timeout: 180000 });

              const responseData = res.data;
              console.log('Backend response for Excel upload:', responseData);

              if (responseData.success) {
              this.messages.push({ role: 'assistant', text: '✅ Excel data processed by backend. Now generating map.' });
              if (responseData.geojsonResult) {
                // 可以处理返回的地理JSON数据
              }
              if (responseData.imageResult) {
                // 可以处理返回的图像数据
              }
              } else {
                this.messages.push({ role: 'assistant', text: '❌ Backend failed to process Excel data: ' + (responseData.message || 'Unknown error') });
              }
            } catch (err) {
            console.error('axios error during Excel upload:', err);
            this.messages.push({ role: 'assistant', text: '❌ Failed to send Excel data to backend: ' + (err.message || err) });
            }
          }
      };

      reader.readAsArrayBuffer(file);
    },

    fetchHistory: async function () {
      try {
        const res = await axios.get(`${API_BASE}/history?limit=20`)
        this.historyList = res.data.history || []
      } catch (e) {
        console.error('获取历史失败', e)
      }
    },

    applyHistory(item) {
      this.userInput = item.text
      this.showHistory = false
      this.sendMessage()
    },

    formatTimestamp(ts) {
      if (!ts) return '未知时间'
      const date = new Date(ts)
      return isNaN(date.getTime()) ? '未知时间' : date.toLocaleString()
    },

    toggleZoom(){
      this.isZoomed = !this.isZoomed
    },

    downloadMap() {
      if (!this.latestMapImage) {
        alert("没有可下载的地图！");
        return;
      }
      this.downloadBase64Image(this.latestMapImage, 'contour_map.png');
    },

    downloadBase64Image(base64Str, filename = 'map.png') {
      let base64Data = base64Str;
      if (base64Str.startsWith('data:')) {
        base64Data = base64Str.split(',')[1];
      }

      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);

      const blob = new Blob([byteArray], { type: 'image/png' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      URL.revokeObjectURL(link.href);
    }
  }
}
</script>

<style scoped>
.page {
  width: 100%;
  display: flex;
  height: 95vh;
  gap: 12px;
  padding: 12px;
  box-sizing: border-box;
}

.qa-card {
  flex: 1;
  max-width: 33%;
  background: #fff;
  border-radius: 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.map-card {
  flex: 2;
  position: relative;
  background: #f0f2f5;
  border-radius: 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  overflow: hidden;
}

/* 标题和按钮同一行，标题居中，按钮在左 */
.header-with-history {
  display: flex;
  align-items: center;
  position: relative;
  margin-bottom: 12px;
}

/* 标题居中 */
.title {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  font-size: 24px;
  color: #333;
  margin: 0;
}

/* 历史按钮在左侧 */
.history-btn {
  padding: 6px 10px;
  border-radius: 50%;
  background: #f0f0f0;
  border: none;
  cursor: pointer;
  z-index: 101;
}

/* 历史侧拉框 */
.history-panel {
  position: absolute;
  top: 0;
  width: 280px;
  background: #fff;
  border-radius:  0 12px 12px 0;
  box-shadow: 0 2px 12px rgba(0,0,0,0.15);
  padding: 12px;
  z-index: 100;
  overflow-y: auto;
}

.messages {
  flex: 1;                  /* 占据剩余空间 */
  /* margin-left: 18rem; */       /* 给左侧历史列表留出空间 */
  overflow-y: auto;         /* 独立滚动 */
}

/* 左侧滑出动画 */
.slide-right-enter-active, .slide-right-leave-active {
  transition: transform 0.3s ease;
}
.slide-right-enter-from, .slide-right-leave-to {
  transform: translateX(-100%); /* 从左侧滑出 */
}

/* 历史任务列表样式 */
.history-item {
  margin-bottom: 8px;
  padding: 6px 8px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}
.history-item:hover {
  background: #f5f5f5;
}

/* 标题字体加粗 */
.history-panel h3 {
  font-weight: bold;
  margin-bottom: 8px;
}

.chat-image img {
  max-width: 200px;
  max-height: 150px;
  border-radius: 8px;
  cursor: pointer;
  margin-top: 4px;
}

.title {
  text-align:center;
  margin-bottom:16px;
  font-size:24px;
  color:#333;
}

.chat-box {
  position: relative;
  display: flex;
  height: calc(100vh - 4rem);
  flex: 1;
  overflow-y: auto;
  background: #fdfdfd;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 10px;
  margin-bottom: 12px;
}

.image-display {
  width: 100%;
  max-height: 300px;
  margin-bottom: 10px;
  border: 1px solid #ddd;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
}

.image-display img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  border-radius: 10px;
}

.input-area {
  display:flex;
  gap:10px;
}

input[type="text"] {
  flex:1;
  padding:10px;
  border-radius:10px;
  border:1px solid #ccc;
  font-size:14px;
}

button {
  padding:10px 16px;
  background:#007bff;
  color:#fff;
  border:none;
  border-radius:10px;
  cursor:pointer;
  font-size:14px;
}

.msg {
  display: flex;
  margin-bottom: 8px;
  word-break: break-word;
  padding: 6px 12px;
  border-radius: 10px;
}


.msg.assistant {
  justify-content: flex-start;
  align-self: flex-start;
}

.msg.user {
  justify-content: flex-end;
  align-self: flex-end;
}

.role-label {
  font-weight: bold;
  margin-right: 6px;
}

.msg.user .role-label {
  order: 1;
}

.msg.user .text {
  order: 0;
}


button:hover{
  background:#0056b3;
}

.text {
  display:inline-block;
  background:#e0f0ff;
  padding:8px 12px;
  border-radius:10px;
  max-width:80%;
  white-space:pre-wrap;
  word-wrap:break-word;
}

.image-wrapper:hover .download-fab{
  opacity:1;
}

.preview-overlay{
  position:fixed;
  top:0;
  left:0;
  right:0;
  bottom:0;
  background:rgba(0,0,0,0.8);
  display:flex;
  align-items:center;
  justify-content:center;
  z-index:9999;
}

.preview-img{
  max-width:90%;
  max-height:90%;
  border-radius:8px;
}

.map-download-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 1001;
  background: #cecece;
  color: #fff;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.map-download-btn:hover {
  background: #585959;
}

.legend-card-vertical {
  position: absolute;
  bottom: 20px;
  right: 20px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
  padding: 20px;
  z-index: 1000;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.legend-title {
  font-weight: bold;
  font-size: 14px;
  margin-bottom: 10px;
}

.legend-content {
  display: flex;
  align-items: stretch;
}

.legend-colors {
  display: flex;
  flex-direction: column;
  gap: 2px; /* 颜色块之间的间距 */
}

.legend-color-box {
  width: 20px;
  height: 20px;
}

.legend-labels-vertical {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  margin-left: 8px;
  font-size: 12px;
  gap: 2px;
}
.legend-labels-vertical span {
  height: 20px;
  line-height: 20px;
  text-align: right;
}

/* 圆形按钮样式 */
.circle-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #007bff;
  color: white;
  border: none;
  font-size: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.circle-btn:hover {
  background: #0056b3;
}

/* 规则上传按钮特殊样式 */
.circle-btn.rules-btn {
  background: #28a745;
}

.circle-btn.rules-btn:hover {
  background: #1e7e34;
}

</style>
