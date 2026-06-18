"""
GeoPredict API — FastAPI Backend
Capacité Portante des Fondations Superficielles
Random Forest Pipeline — Eurocode 7
Université Badji Mokhtar Annaba — Master 2 Géotechnique
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import pandas as pd
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
#  Helper — compute Rw1, Rw2, gamma1av, gamma2av
#  Selon le cours Pr. Sbartai — Chapitre 2, pp. 9-10
# ─────────────────────────────────────────────
def compute_water_corrections(inp: PredictionInput):
    """
    Correction nappe phréatique selon Pr. Sbartai (cours Fondations Superficielles)

    CAS 0 : Zw >= Df + B  →  aucune influence
    CAS 1 : Zw <= Df      →  nappe au-dessus de la base de la fondation
             zw1 = Zw_m (profondeur nappe depuis surface)
             Rw1 = ½(1 + zw1/D)
             Rw2 = ½(1 + zw2/B)  avec zw2=0 (nappe au niveau fondation)
             γ1av = (γ1·zw1 + γ1sat·(D-zw1)) / D

    CAS 2 : Df < Zw < Df+B  →  nappe sous la fondation, dans zone d'influence B
             zw1 = Df (nappe en dessous → terme Nq non affecté → Rw1=1)
             zw2 = Zw_m - Df (distance nappe sous la fondation)
             Rw1 = ½(1 + zw1/D) = ½(1 + D/D) = 1.0
             Rw2 = ½(1 + zw2/B)
             γ2av = (γ2·zw2 + γ2sat·(B-zw2)) / B
    """
    gamma_w   = 9.81                        # kN/m³
    gamma_sat = inp.gamma_sat               # γsat
    gamma_nat = inp.gamma_kNm3             # γ naturel
    gamma_dej = gamma_sat - gamma_w        # γ' déjaugé = γsat - γw
    D  = inp.Df_m
    Zw = inp.Zw_m
    B  = inp.B_m if inp.B_m > 0 else (2.0 * inp.R_m if inp.R_m > 0 else 1.0)

    if inp.cas_nappe == 0:
        # ── Cas 0 : aucune influence ──
        Rw1      = 1.0
        Rw2      = 1.0
        gamma1av = gamma_nat   # γ naturel pour terme Nq
        gamma2av = gamma_nat   # γ naturel pour terme Nγ

    elif inp.cas_nappe == 1:
        # ── Cas 1 : nappe au-dessus ou au niveau de la fondation (Zw <= Df) ──
        # Pr. Sbartai p.9 :
        #   zw1 = Zw (profondeur nappe depuis surface)
        #   zw2 = 0  (nappe est au niveau ou au-dessus de la base → sol saturé sous fondation)
        #   Rw1 = ½(1 + zw1/D)
        #   Rw2 = ½(1 + zw2/B) = ½(1 + 0/B) = 0.5
        #   γ1av = (γnat·zw1 + γsat·(D - zw1)) / D
        zw1 = min(Zw, D)                           # nappe depuis surface (clampé à D)
        zw2 = 0.0                                   # sol sous fondation entièrement saturé
        Rw1 = 0.5 * (1.0 + zw1 / D)
        Rw2 = 0.5 * (1.0 + zw2 / B)               # = 0.5
        gamma1av = (gamma_nat * zw1 + gamma_sat * (D - zw1)) / D
        gamma2av = gamma_dej                        # sol sous fondation = γ déjaugé

    else:
        # ── Cas 2 : nappe sous la fondation, dans la zone B (Df < Zw < Df+B) ──
        # Pr. Sbartai p.10 :
        #   zw1 = D (nappe sous fondation → terme surcharge non affecté)
        #   zw2 = Zw - Df (distance nappe depuis base fondation)
        #   Rw1 = ½(1 + zw1/D) = ½(1 + 1) = 1.0
        #   Rw2 = ½(1 + zw2/B)
        #   γ2av = (γnat·zw2 + γsat·(B - zw2)) / B
        zw1 = D                                     # → Rw1 = 1.0
        zw2 = min(Zw - D, B)                       # distance nappe sous fondation (clampé à B)
        Rw1 = 0.5 * (1.0 + zw1 / D)               # = 1.0
        Rw2 = 0.5 * (1.0 + zw2 / B)
        gamma1av = gamma_nat                        # sol au-dessus fondation = γ naturel
        gamma2av = (gamma_nat * zw2 + gamma_dej * (B - zw2)) / B

    # gamma_eff_kNm3 : valeur représentative pour les features RF (moyenne pondérée)
    gamma_eff = (gamma1av + gamma2av) / 2.0

    return Rw1, Rw2, gamma_eff, gamma1av, gamma2av


# ─────────────────────────────────────────────
#  Prediction endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(inp: PredictionInput):
    if model_stage1 is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")

    try:
        # ── Water corrections (Pr. Sbartai) ──
        Rw1, Rw2, gamma_eff, gamma1av, gamma2av = compute_water_corrections(inp)

        # ── Stage 1 features ──
        X1 = pd.DataFrame([{
            "c_kPa":         inp.c_kPa,
            "phi_deg":       inp.phi_deg,
            "gamma_kNm3":    inp.gamma_kNm3,
            "gamma_sat":     inp.gamma_sat,
            "B_m":           inp.B_m,
            "L_m":           inp.L_m,
            "R_m":           inp.R_m,
            "Df_m":          inp.Df_m,
            "Zw_m":          inp.Zw_m,
            "cas_nappe":     inp.cas_nappe,
            "Rw1":           Rw1,
            "Rw2":           Rw2,
            "gamma_eff_kNm3": gamma_eff,
            "Type_fond":     inp.Type_fond,
        }])

        stage1_pred = model_stage1.predict(X1)[0]
        # Outputs: Nq, Nc, Ng, sq, sc, sg
        Nq_p, Nc_p, Ng_p, sq_p, sc_p, sg_p = [float(v) for v in stage1_pred]

        # ── Stage 2 — qu & q_adm features ──
        X2 = pd.DataFrame([{
            "c_kPa":          inp.c_kPa,
            "phi_deg":        inp.phi_deg,
            "gamma_kNm3":     inp.gamma_kNm3,
            "gamma_sat":      inp.gamma_sat,
            "B_m":            inp.B_m,
            "L_m":            inp.L_m,
            "R_m":            inp.R_m,
            "Df_m":           inp.Df_m,
            "Zw_m":           inp.Zw_m,
            "cas_nappe":      inp.cas_nappe,
            "Rw1":            Rw1,
            "Rw2":            Rw2,
            "gamma_eff_kNm3": gamma_eff,
            "q_app_kPa":      inp.q_app_kPa,
            "Nq_pred":        Nq_p,
            "Nc_pred":        Nc_p,
            "Ng_pred":        Ng_p,
            "sq_pred":        sq_p,
            "sc_pred":        sc_p,
            "sg_pred":        sg_p,
            "Type_fond":      inp.Type_fond,
        }])

        stage2_pred = model_stage2.predict(X2)[0]
        qu_p, qadm_p = float(stage2_pred[0]), float(stage2_pred[1])

        # ── Stage 2 — FS ──
        X3 = pd.DataFrame([{
            "qu_pred":    qu_p,
            "q_app_kPa":  inp.q_app_kPa,
        }])
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
                "Rw1":            round(Rw1,      4),
                "Rw2":            round(Rw2,      4),
                "gamma_eff_kNm3": round(gamma_eff, 4),
                "gamma1av":       round(gamma1av,  4),
                "gamma2av":       round(gamma2av,  4),
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
