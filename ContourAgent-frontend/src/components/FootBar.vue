<template>
  <footer class="bg-black text-white py-3">
    <div class="container">
      <div class="d-flex flex-column flex-md-row align-items-center justify-content-between">
        <!-- 左侧菜单 -->
        <nav class="mb-2 mb-md-0">
          <ul class="nav">
            <li class="nav-item" v-for="item in menu" :key="item.name">
              <a class="nav-link px-2 text-white" :href="item.link">{{ item.name }}</a>
            </li>
          </ul>
        </nav>
        <!-- 分隔竖线（仅大屏显示）-->
<div class="mx-3 d-none d-md-block" style="height: 36px; border-left: 2px solid #555;"></div>        <!-- 右侧版权声明 -->
<div class="d-flex flex-column align-items-center">
  <span>&copy; 2025. 版权所有</span>
  <span style="font-size: 0.95em; color: #ccc; margin-top: 4px;">湖北省武汉市蔡甸区蔡甸街大学路111号 长江大学地球科学学院</span>
</div>
      </div>
    </div>
  </footer>
</template>

<script setup>
const menu = [
  { name: '官网首页', link: '#/HomeView' },
  { name: '古地理文献', link: '#/Literature' },
  { name: '测井数据', link: '#/WellDataBase' },
  { name: '分析化验', link: '#/Chemistry' },
  { name: '地图服务', link: '#/MapService' },
  { name: '联系我们', link: '#/ContactUs' },
  { name: '帮助文档', link: '#/HelpFile' },
];
</script>

<style scoped>
/* 可选：调整分隔线高度（适配内容高度） */
@media (max-width: 767.98px) {
  .mx-3.d-none.d-md-block {
    display: none !important;
  }
}

.nav-link:hover {
  color: #797878 !important; /* 优雅的高亮色 */
  text-decoration: underline;
  transition: color 0.2s;
}
</style>
