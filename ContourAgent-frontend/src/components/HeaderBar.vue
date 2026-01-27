<template>
  <nav class="navbar navbar-expand-lg navbar-dark">
    <div class="container-fluid">
      <div class="row w-100 align-items-center">
        <div class="col-auto d-flex justify-content-center align-items-center" style="height: 100%;">
          <a class="navbar-brand custom-brand mx-auto" href="#/HomeView">岩相古地理知识库</a>
      </div>
        <div class="col">
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav custom-nav">
          <li class="nav-item">

            <a class="nav-link" href="#/Literature">古地理文献</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#/WellDataBase">测井数据</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#/Chemistry">分析化验数据</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#/MapService">地图服务</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#/ChatView">
<!--              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-slack" viewBox="0 0 16 16">-->
<!--                <path d="M3.362 10.11c0 .926-.756 1.681-1.681 1.681S0 11.036 0 10.111C0 9.186.756 8.43 1.68 8.43h1.682v1.68zm.846 0c0-.924.756-1.68 1.681-1.68s1.681.756 1.681 1.68v4.21c0 .924-.756 1.68-1.68 1.68a1.685 1.685 0 0 1-1.682-1.68v-4.21zM5.89 3.362c-.926 0-1.682-.756-1.682-1.681S4.964 0 5.89 0s1.68.756 1.68 1.68v1.682H5.89zm0 .846c.924 0 1.68.756 1.68 1.681S6.814 7.57 5.89 7.57H1.68C.757 7.57 0 6.814 0 5.89c0-.926.756-1.682 1.68-1.682h4.21zm6.749 1.682c0-.926.755-1.682 1.68-1.682.925 0 1.681.756 1.681 1.681s-.756 1.681-1.68 1.681h-1.681V5.89zm-.848 0c0 .924-.755 1.68-1.68 1.68A1.685 1.685 0 0 1 8.43 5.89V1.68C8.43.757 9.186 0 10.11 0c.926 0 1.681.756 1.681 1.68v4.21zm-1.681 6.748c.926 0 1.682.756 1.682 1.681S11.036 16 10.11 16s-1.681-.756-1.681-1.68v-1.682h1.68zm0-.847c-.924 0-1.68-.755-1.68-1.68 0-.925.756-1.681 1.68-1.681h4.21c.924 0 1.68.756 1.68 1.68 0 .926-.756 1.681-1.68 1.681h-4.21z"/>-->
<!--              </svg>-->
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-twitch" viewBox="0 0 16 16">
                <path d="M3.857 0 1 2.857v10.286h3.429V16l2.857-2.857H9.57L14.714 8V0H3.857zm9.714 7.429-2.285 2.285H9l-2 2v-2H4.429V1.143h9.142v6.286z"/>
                <path d="M11.857 3.143h-1.143V6.57h1.143V3.143zm-3.143 0H7.571V6.57h1.143V3.143z"/>
              </svg>
              智能问答</a>
          </li>
<!--          <li class="nav-item">-->
<!--            <a class="nav-link" href="#/Mapping">克里金插值</a>-->
<!--          </li>-->
          <li class="nav-item">
            <a class="nav-link" href="#/AgentMapping">智能制图</a>
          </li>

<!--          <li class="nav-item">-->
<!--            <a class="nav-link" href="#/ChartTest">测井曲线测试</a>-->
<!--          </li>-->
<!--          <li class="nav-item">-->
<!--            <a class="nav-link" href="#/Mapping">克里金插值</a>-->
<!--          </li>-->
<!--          <li class="nav-item">-->
<!--            <a class="nav-link" href="#/Neo4jTest">Neo4j测试</a>-->
<!--          </li>-->
<!--          <li class="nav-item">-->
<!--            <a class="nav-link" href="#/LiteratureKnowledgeGraph">知识图谱组件测试</a>-->
<!--          </li>-->
        </ul>
          </div>
        </div>
        <div class="col-auto ms-auto d-flex align-items-center">
          <!--          <el-avatar :size="24" icon="el-icon-user" style="margin-right:8px;" />-->
          <!--          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-person-bounding-box" viewBox="0 0 16 16" style="margin-right:8px;">-->
          <!--            <path d="M1.5 1a.5.5 0 0 0-.5.5v3a.5.5 0 0 1-1 0v-3A1.5 1.5 0 0 1 1.5 0h3a.5.5 0 0 1 0 1h-3zM11 .5a.5.5 0 0 1 .5-.5h3A1.5 1.5 0 0 1 16 1.5v3a.5.5 0 0 1-1 0v-3a.5.5 0 0 0-.5-.5h-3a.5.5 0 0 1-.5-.5zM.5 11a.5.5 0 0 1 .5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 1 0 1h-3A1.5 1.5 0 0 1 0 14.5v-3a.5.5 0 0 1 .5-.5zm15 0a.5.5 0 0 1 .5.5v3a1.5 1.5 0 0 1-1.5 1.5h-3a.5.5 0 0 1 0-1h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 1 .5-.5z"/>-->
          <!--            <path d="M3 14s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H3zm8-9a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/>-->
          <!--          </svg>-->
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-person-workspace " viewBox="0 0 16 16" style="margin-right:8px;">>
            <path d="M4 16s-1 0-1-1 1-4 5-4 5 3 5 4-1 1-1 1H4Zm4-5.95a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"/>
            <path d="M2 1a2 2 0 0 0-2 2v9.5A1.5 1.5 0 0 0 1.5 14h.653a5.373 5.373 0 0 1 1.066-2H1V3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v9h-2.219c.554.654.89 1.373 1.066 2h.653a1.5 1.5 0 0 0 1.5-1.5V3a2 2 0 0 0-2-2H2Z"/>
          </svg>
          <a class="nav-link" href="#/ManagementData">后台管理</a>
          <div style="width: 10px;"></div> <!-- 添加一个间隔 -->
          <a class="nav-link" href="#/">退出登录</a>

        </div>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'HeaderBar',
  data(){
    return {
      isContactActive: false,
      isHomeActive: false,
      isHelpActive: false,
      isDataOverviewActive: false,
      isLoginActive: false,
      isWellDataBaseActive: false,
      selectedSubId: this.$route.query.selectedSubId ,  // 从路由中获取的 selectedSubId
    }
  },

  watch: {
    // 监听路由变化，更新激活状态
    '$route'(to, from) {
      if (to.meta.activeTab) {
        this.setActive(to.meta.activeTab);
      }
    }
  },
  methods: {
    setActive(tab) {
      const tabs = {
        'contact': 'isContactActive',
        'home': 'isHomeActive',
        'help': 'isHelpActive',
        'DataOverview': 'isDataOverviewActive',
        'login': 'isLoginActive',
        'well': 'isWellDataBaseActive',
      };
      if (tabs[tab]) {
        this[tabs[tab]] = true;
      }
    },
    navigateToManagementData(subid) {
      const currentRoute = this.$route.name;
      this.selectedSubId = subid;
      // 如果当前路由是 'ManagementData'，更新 query 参数
      if (currentRoute === 'ManagementData') {
        this.$router.replace({
          name: 'ManagementData',
          query: { selectedSubId: subid },
          hash: Date.now().toString() // 使用时间戳强制刷新组件
        });
      } else {
        // 如果当前路由不是 'ManagementData'，直接跳转
        this.$router.push({ name: 'ManagementData', query: { selectedSubId: subid } });
      }
    },
    gotoDisjointedbodyDetails () {
      this.$router.push({ name: 'DisjointedbodyDetails' })
    },
    intoBackstage () {
      this.$router.push({ name: 'DataManagement' })
    },
    intoManagement () {
      this.$router.push({ name: 'PermissionManagement' })
    },
    logout1 () {
      // 使用 replace 替换当前的历史记录条目，并跳转到登录页面
      // 使用 window.location.replace 替换当前页面并跳转到登录页面
      window.location.replace('/');  // 登录页面的 URL
    }
  },
  created() {
    this.setActive(this.$route.meta.activeTab);
  }
}
</script>

<style scoped>
.navbar {
  background-color: #262424 !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important;

  background-image: url('../assets/black-stone.jpg');
  background-repeat: no-repeat;
  background-position: center center;
  background-size: cover;

  min-height: 55px;
}


.custom-nav {
  margin-left: 0;
}


.navbar,
.navbar .navbar-brand,
.navbar .nav-link {
  color: #fff !important;
}

.navbar .nav-link.active,
.navbar .nav-link:hover {
  color: #f16d08 !important;
}

#navbarNav {

}

</style>
