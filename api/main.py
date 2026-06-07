"""
GeoPredict API — FastAPI Backend
Capacité Portante des Fondations Superficielles
Random Forest Pipeline — Eurocode 7
Université Badji Mokhtar Annaba — Master 2 Géotechnique
Sous la direction du Pr. Sbartai
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
#  App init
# ─────────────────────────────────────────────
app = FastAPI(
    title="GeoPredict API",
    description="Prédiction de capacité portante par Random Forest — Eurocode 7",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model_stage1   = joblib.load(os.path.join(BASE_DIR, "RF_Stage1_model.pkl"))
    model_stage2   = joblib.load(os.path.join(BASE_DIR, "RF_Stage2_qu_qadm.pkl"))
    model_stage2fs = joblib.load(os.path.join(BASE_DIR, "RF_Stage2_FS.pkl"))
    print("✅ Modèles chargés avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèles : {e}")
    model_stage1 = model_stage2 = model_stage2fs = None


# ─────────────────────────────────────────────
#  Input schema
# ─────────────────────────────────────────────
class PredictionInput(BaseModel):
    c_kPa:       float = Field(..., ge=0,   description="Cohésion c' (kPa)")
    phi_deg:     float = Field(..., ge=0, le=50, description="Angle de frottement φ' (°)")
    gamma_kNm3:  float = Field(..., gt=0,   description="Poids volumique naturel γ (kN/m³)")
    gamma_sat:   float = Field(..., gt=0,   description="Poids volumique saturé γsat (kN/m³)")
    B_m:         float = Field(0.0, ge=0,   description="Largeur B (m) — 0 si circulaire")
    L_m:         float = Field(0.0, ge=0,   description="Longueur L (m) — 0 si circulaire")
    R_m:         float = Field(0.0, ge=0,   description="Rayon R (m) — 0 si rectangulaire")
    Df_m:        float = Field(..., gt=0,   description="Profondeur d'encastrement Df (m)")
    Zw_m:        float = Field(..., ge=0,   description="Profondeur nappe phréatique Zw (m)")
    cas_nappe:   int   = Field(..., ge=0, le=2, description="Cas nappe : 0=aucun, 1=au-dessus Df, 2=dans zone B")
    q_app_kPa:   float = Field(..., gt=0,   description="Charge appliquée q_app (kPa)")
    Type_fond:   Literal["Rectangulaire", "Carrée", "Circulaire"] = Field(..., description="Type de fondation")

    class Config:
        json_schema_extra = {
            "example": {
                "c_kPa": 16.87,
                "phi_deg": 22.01,
                "gamma_kNm3": 18.75,
                "gamma_sat": 21.24,
                "B_m": 1.52,
                "L_m": 2.0,
                "R_m": 0.0,
                "Df_m": 1.58,
                "Zw_m": 1.02,
                "cas_nappe": 1,
                "q_app_kPa": 131.82,
                "Type_fond": "Rectangulaire"
            }
        }


# ─────────────────────────────────────────────
#  Helper — compute Rw1, Rw2, gamma_eff
# ─────────────────────────────────────────────
def compute_water_corrections(inp: PredictionInput):
    """
    Rw1, Rw2, gamma_eff selon Eurocode 7 / Das (2011)
    cas=0 : pas d'influence
    cas=1 : nappe au-dessus de la fondation
    cas=2 : nappe dans la zone d'influence B
    """
    gamma_w = 9.81  # kN/m³
    gamma_eff_sat = inp.gamma_sat - gamma_w  # poids déjaugé

    if inp.cas_nappe == 0:
        Rw1 = 1.0
        Rw2 = 1.0
        gamma_eff = inp.gamma_kNm3
    elif inp.cas_nappe == 1:
        # Nappe au-dessus ou au niveau de la fondation
        Rw1 = 0.5
        Rw2 = 0.5
        gamma_eff = gamma_eff_sat
    else:
        # cas=2 : nappe dans la zone d'influence (Df < Zw < Df+B)
        B_ref = inp.B_m if inp.B_m > 0 else (2 * inp.R_m if inp.R_m > 0 else 1.0)
        zw_rel = inp.Zw_m - inp.Df_m  # distance nappe sous la fondation
        ratio  = min(max(zw_rel / B_ref, 0.0), 1.0)
        Rw1 = 0.5 + 0.5 * ratio
        Rw2 = 0.5 + 0.5 * ratio
        gamma_eff = inp.gamma_kNm3 * ratio + gamma_eff_sat * (1 - ratio)

    return Rw1, Rw2, gamma_eff


# ─────────────────────────────────────────────
#  Prediction endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(inp: PredictionInput):
    if model_stage1 is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")

    try:
        # ── Water corrections ──
        Rw1, Rw2, gamma_eff = compute_water_corrections(inp)

        # ── Stage 1 features ──
        # Order: c_kPa, phi_deg, gamma_kNm3, gamma_sat, B_m, L_m, R_m,
        #        Df_m, Zw_m, cas_nappe, Rw1, Rw2, gamma_eff_kNm3, Type_fond
        X1 = np.array([[
            inp.c_kPa, inp.phi_deg, inp.gamma_kNm3, inp.gamma_sat,
            inp.B_m, inp.L_m, inp.R_m,
            inp.Df_m, inp.Zw_m, inp.cas_nappe,
            Rw1, Rw2, gamma_eff,
            inp.Type_fond           # sklearn pipeline handles encoding
        ]], dtype=object)

        stage1_pred = model_stage1.predict(X1)[0]
        # Outputs: Nq, Nc, Ng, sq, sc, sg
        Nq_p, Nc_p, Ng_p, sq_p, sc_p, sg_p = [float(v) for v in stage1_pred]

        # ── Stage 2 — qu & q_adm features ──
        # Order: c_kPa, phi_deg, gamma_kNm3, gamma_sat, B_m, L_m, R_m,
        #        Df_m, Zw_m, cas_nappe, Rw1, Rw2, gamma_eff_kNm3, q_app_kPa,
        #        Nq_pred, Nc_pred, Ng_pred, sq_pred, sc_pred, sg_pred, Type_fond
        X2 = np.array([[
            inp.c_kPa, inp.phi_deg, inp.gamma_kNm3, inp.gamma_sat,
            inp.B_m, inp.L_m, inp.R_m,
            inp.Df_m, inp.Zw_m, inp.cas_nappe,
            Rw1, Rw2, gamma_eff,
            inp.q_app_kPa,
            Nq_p, Nc_p, Ng_p, sq_p, sc_p, sg_p,
            inp.Type_fond
        ]], dtype=object)

        stage2_pred = model_stage2.predict(X2)[0]
        qu_p, qadm_p = float(stage2_pred[0]), float(stage2_pred[1])

        # ── Stage 2 — FS ──
        # Features: qu_pred, q_app_kPa
        X3 = np.array([[qu_p, inp.q_app_kPa]])
        FS_p = float(model_stage2fs.predict(X3)[0])

        # ── Safety status (from Excel legend) ──
        if FS_p >= 3.2:
            status = "Très sécuritaire"
            status_en = "Very safe"
            status_ar = "آمن جداً"
            color = "green"
        elif FS_p >= 2.5:
            status = "Courant"
            status_en = "Normal"
            status_ar = "عادي"
            color = "blue"
        elif FS_p >= 2.0:
            status = "Défavorable"
            status_en = "Unfavorable"
            status_ar = "غير ملائم"
            color = "yellow"
        else:
            status = "Sous-dimensionné"
            status_en = "Undersized"
            status_ar = "غير كافٍ"
            color = "red"

        return {
            "success": True,
            "inputs": {
                "Rw1": round(Rw1, 4),
                "Rw2": round(Rw2, 4),
                "gamma_eff_kNm3": round(gamma_eff, 4),
            },
            "stage1": {
                "Nq": round(Nq_p, 4),
                "Nc": round(Nc_p, 4),
                "Ng": round(Ng_p, 4),
                "sq": round(sq_p, 4),
                "sc": round(sc_p, 4),
                "sg": round(sg_p, 4),
            },
            "stage2": {
                "qu_kPa":   round(qu_p,   2),
                "q_adm_kPa": round(qadm_p, 2),
                "FS":        round(FS_p,   3),
            },
            "verdict": {
                "status":    status,
                "status_en": status_en,
                "status_ar": status_ar,
                "color":     color,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


# ─────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "app":     "GeoPredict API",
        "version": "1.0.0",
        "status":  "online",
        "models":  "loaded" if model_stage1 else "not loaded",
        "doc":     "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_ready": model_stage1 is not None}
