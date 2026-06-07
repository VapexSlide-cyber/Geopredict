# 🏗️ GeoPredict — Guide de Déploiement Complet

**Capacité Portante par Random Forest — Eurocode 7**  
Université Badji Mokhtar Annaba · Master 2 Géotechnique · Pr. Sbartai

---

## 📁 Structure du projet

```
geopredict/
├── api/
│   ├── main.py                  ← API FastAPI
│   ├── requirements.txt         ← Dépendances Python
│   ├── render.yaml              ← Config Render.com
│   ├── RF_Stage1_model.pkl      ← Copiez vos .pkl ici
│   ├── RF_Stage2_qu_qadm.pkl
│   └── RF_Stage2_FS.pkl
└── web/
    └── index.html               ← Site web complet
```

---

## 🚀 ÉTAPE 1 — Préparer GitHub

### 1.1 Créer le repo
```bash
git init geopredict
cd geopredict
```

### 1.2 Copier les fichiers .pkl dans api/
```bash
cp votre_dossier/RF_Stage1_model.pkl      api/
cp votre_dossier/RF_Stage2_qu_qadm.pkl   api/
cp votre_dossier/RF_Stage2_FS.pkl        api/
```

### 1.3 Push sur GitHub
```bash
git add .
git commit -m "Initial GeoPredict"
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/geopredict.git
git push -u origin main
```

---

## 🌐 ÉTAPE 2 — Déployer l'API sur Render.com

1. Aller sur [render.com](https://render.com) → **New Web Service**
2. Connecter votre repo GitHub `geopredict`
3. Configurer :
   - **Root Directory** : `api`
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan** : Free
4. Cliquer **Create Web Service**
5. Attendre 5-10 min → noter votre URL (ex: `https://geopredict-api.onrender.com`)

---

## 🌍 ÉTAPE 3 — Mettre à jour le site web

Dans `web/index.html`, ligne ~690, remplacer :
```javascript
const API_URL = "https://YOUR-APP.onrender.com";
```
par votre vraie URL Render :
```javascript
const API_URL = "https://geopredict-api.onrender.com";
```

---

## ⚡ ÉTAPE 4 — Déployer le site sur Vercel

1. Aller sur [vercel.com](https://vercel.com) → **Add New Project**
2. Connecter votre repo GitHub
3. Configurer :
   - **Root Directory** : `web`
   - **Framework** : Other (HTML statique)
4. Cliquer **Deploy**
5. Votre site est en ligne sur `https://geopredict.vercel.app` 🎉

---

## 🎯 Personnalisation (à faire)

Dans `web/index.html`, section "ÉTUDIANTS DÉVELOPPEURS" :
```html
<div id="student-names">
  Votre Prénom NOM — Étudiant Master 2 Géotechnique<br>
  Prénom NOM — Étudiant Master 2 Géotechnique
</div>
```

---

## ✅ Test local de l'API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
# → Ouvrir http://localhost:8000/docs
```

---

## 📊 Performance des modèles

| Modèle | R² Test | R² CV |
|--------|---------|-------|
| Stage 1 — Nq/Nc/Nγ | 0.9638 | 0.9537 |
| Stage 2 — qu/q_adm | 0.9575 | 0.9578 |
| Stage 2 — FS | **0.9659** | 0.9659 |

---

*Généré par GeoPredict · Université Badji Mokhtar Annaba*
