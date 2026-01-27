import { createRouter, createWebHistory } from 'vue-router'
import AgentMapping from '../views/AgentMapping.vue'

const routes = [
    {
        path: '/',
        component: AgentMapping
    }
]

export default createRouter({
    history: createWebHistory(),
    routes
})


// import { createRouter, createWebHashHistory } from 'vue-router'
// //"npm install vue-router@4"
// import Login from '../views/Login.vue'
// import HelloWorld from '../views/HelloWorld.vue'
// const routes = [
//     {
//         path: '/',
//         name: 'home',
//         component: Login
//     },
//     {
//         path: '/HomeView',
//         name: 'HomeView',
//         component: () => import(/* webpackChunkName: "HomeView" */ '../views/HomeView.vue')
//     },
//     {
//         path: '/ContactUs',
//         name: 'ContactUs',
//         component: () => import(/* webpackChunkName: "ContactUs" */ '../views/ContactUs.vue'),
//         meta:{ activeTab: 'contact'}
//     },
//     {
//         path: '/HelpFile',
//         name: 'HelpFile',
//         component: () => import(/* webpackChunkName: "HelpFile" */ '../views/HelpFile.vue'),
//         meta:{ activeTab: 'about'}
//     },
//     {
//         path: '/Chemistry',
//         name: 'Chemistry',
//         component: () => import(/* webpackChunkName: "Chemistry" */ '../views/Chemistry.vue'),
//         meta:{ activeTab: 'chemistry'}
//     },
//     {
//         path: "/LiteratureDetails/:id",
//         name: "LiteratureDetails",
//         component: () => import(/* webpackChunkName: "LiteratureDetails" */ "../views/LiteratureDetails.vue"),
//         meta: { activeTab: "literature" }
//     },
//
//     {
//         path: '/ManagementData',
//         name: 'ManagementData',
//         component: () => import(/* webpackChunkName: "ManagementData" */ '../views/ManagementData.vue'),
//         meta:{ activeTab: 'data'}
//     },
//     {
//         path: '/Literature',
//         name: 'Literature',
//         component: () => import(/* webpackChunkName: "LoggingCase" */ '../views/Literature.vue'),
//         meta:{ activeTab: 'data'}
//     },
//     {
//         path: '/MapService',
//         name: 'MapService',
//         component: () => import(/* webpackChunkName: "LoggingCase" */ '../views/MapService.vue'),
//         meta:{ activeTab: 'data'}
//     },
//     {
//         path : '/Register',
//         name : 'Register',
//         component : () => import(/* webpackChunkName: "Register" */ '../views/Register.vue')
//     },
//     {
//         path: '/RetrievePassword',
//         name: 'RetrievePassword',
//         component: () => import(/* webpackChunkName: "RetrievePassword" */ '../views/RetrievePassword.vue')
//     },
//     {
//         path : '/UserAgreement',
//         name : 'UserAgreement',
//         component : () => import(/* webpackChunkName: "UserAgreement" */ '../views/UserAgreement.vue')
//     },
//     {
//         path : '/WellDataBase',
//         name : 'WellDataBase',
//         component : () => import(/* webpackChunkName: "WellDataBase" */ '../views/WellDataBase.vue')
//     },
//     {
//         path : '/WellDetail',
//         name : 'WellDetail',
//         component : () => import(/* webpackChunkName: "WellDetail" */ '../views/WellDetail.vue')
//     },
//     {
//         path : '/ChatView',
//         name : 'ChatView',
//         component : () => import(/* webpackChunkName: "ChatView" */ '../views/ChatView.vue')
//     },
//     {
//         path : '/ChartTest',
//         name : 'ChartTest',
//         component : () => import(/* webpackChunkName: "ChatView" */ '../views/ChartTest.vue')
//     },
//     {
//         path : '/Mapping',
//         name : 'Mapping',
//         component : () => import(/* webpackChunkName: "ChatView" */ '../views/Mapping.vue')
//     },
//     {
//         path : '/AgentMapping',
//         name : 'AgentMapping',
//         component : () => import(/* webpackChunkName: "ChatView" */ '../views/AgentMapping.vue')
//     },
//     {
//         path : '/Neo4jTest',
//         name : 'Neo4jTest',
//         component : () => import(/* webpackChunkName: "Neo4jTest" */ '../views/Neo4jTest.vue')
//     },
//     {
//         path:'/LiteratureKnowledgeGraph',
//         name:'LiteratureKnowledgeGraph',
//         component:()=>import(/*webpackChunkName: "literatureKnowledgeGraph" */ '../views/LiteratureKnowledgeGraph.vue')
//     },
//     {
//         path: '/HomeNeo4j',
//         name: 'HomeNeo4j',
//         component: () => import(/* webpackChunkName: "HomeNeo4j" */ '../views/HomeNeo4j.vue')
//     },
//     {
//         path: '/WellKnowledgeGraph',
//         name: 'WellKnowledgeGraph',
//         component: () => import(/* webpackChunkName: "WellKnowledgeGraph" */ '../views/WellKnowledgeGraph.vue')
//     }
//
// ]
//
//
// const router = createRouter({
//     history: createWebHashHistory(),
//     routes,
//     scrollBehavior(to, from, savedPosition) {
//         return { top: 0 };
//     }
// })
//
// export default router
