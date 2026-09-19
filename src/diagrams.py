PIPELINE_DOT = """
digraph Pipeline {
    rankdir=LR;
    node [shape=box, style=filled, fontname="Inter", fillcolor="#FFFFFF", color="#1F5C8B", penwidth=1.5];
    edge [color="#4A5560"];
    
    subgraph cluster_acq {
        label="ACQUISITION";
        fontname="Inter";
        color="#DDE2E7";
        acq_vib [label="4-ch accelerometer\n(100 kHz)"];
        acq_aud [label="Zoom H5 recorder\n(44.1 kHz)"];
        acq_vid [label="iOS + Android video\n(not consumed by model)", style="filled,dashed", fillcolor="#F5F7F9"];
    }

    subgraph cluster_etl {
        label="ETL (Policy-hash sealed)";
        fontname="Inter";
        color="#DDE2E7";
        etl_vib [label="QC & Jitter check\nresample_poly 100k->25k", fillcolor="#E8F4F8"];
        etl_aud [label="FLAC (lossless)", fillcolor="#E8F4F8"];
        etl_vid [label="WebP frames", style="filled,dashed", fillcolor="#F5F7F9"];
        storage [label="TAR shards +\nparquet index"];
    }

    subgraph cluster_feat {
        label="FEATURES";
        fontname="Inter";
        color="#DDE2E7";
        feat_vib [label="Vibration windowing (1.0s/0.5s hop)\ntime/spectral/env/cross-corr\nClip aggr (mean,std,min,max,med,p90)"];
        feat_aud [label="Audio windowing\nspectral/band/MFCC\nClip aggr"];
    }

    subgraph cluster_models {
        label="MODELS";
        fontname="Inter";
        color="#DDE2E7";
        mod_vib [label="Vib clip 885 feats ->\nXGB Binary & Multiclass"];
        mod_aud [label="Aud clip 600 feats ->\nXGB Binary"];
        mod_fuse [label="Fused 1486 feats ->\n5-fold XGB ensemble"];
        mod_win [label="Fused windows ->\nXGB window model"];
    }

    subgraph cluster_decision {
        label="DECISION";
        fontname="Inter";
        color="#DDE2E7";
        decision [label="p_fault(t) ->\n{Moving Avg | EMA | N-of-M | Persistence}", fillcolor="#FDE8E8", color="#C2410C"];
        alarm [label="Alarm State", fillcolor="#C2410C", fontcolor="white"];
    }

    acq_vib -> etl_vib;
    acq_aud -> etl_aud;
    acq_vid -> etl_vid [style=dashed];
    
    etl_vib -> storage;
    etl_aud -> storage;
    etl_vid -> storage [style=dashed];
    
    storage -> feat_vib;
    storage -> feat_aud;
    
    feat_vib -> mod_vib;
    feat_aud -> mod_aud;
    
    feat_vib -> mod_fuse;
    feat_aud -> mod_fuse;
    
    mod_fuse -> mod_win;
    mod_win -> decision;
    decision -> alarm;
}
"""

FUSION_DOT = """
digraph Fusion {
    rankdir=LR;
    node [shape=box, style=filled, fontname="Inter", fillcolor="#FFFFFF", color="#1F5C8B", penwidth=1.5];
    
    vib_feat [label="Vibration Clip Features\n(885)"];
    aud_feat [label="Audio Clip Features\n(600)"];
    cross_feat [label="Cross-modal ratio\n(1)"];
    
    concat [label="Feature-level concatenation\n(1486 features)", fillcolor="#E8F4F8"];
    ensemble [label="5-fold XGBoost Ensemble"];
    prob [label="p_fault"];
    
    vib_feat -> concat;
    aud_feat -> concat;
    cross_feat -> concat;
    
    concat -> ensemble;
    ensemble -> prob;
}
"""
