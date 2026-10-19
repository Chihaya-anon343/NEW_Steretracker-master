#!/usr/bin/env python3
"""Physical planar stress suite. Entry: generate_approach_sequence.py --suite-config ...

R maps target (+X right, +Y down, +Z cross product) to camera coordinates.
Yaw/pitch/roll retain the original script's Rz @ Rx @ Ry convention.
Full-precision GT uses class0 annotation bbox center as its origin. The existing
tracker's class scales and manually configured class1 offset are not changed.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from generate_synthetic_dataset import load_points, project_points, render, rotation_matrix, write_points


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trajectory(kind, i, count):
    s = i / max(count - 1, 1)
    if kind == "reference":
        return (min(20 * 1.06 ** i, 2600), 2.5 * i,
                8 * np.sin(2*np.pi*i/40), 8*(np.cos(2*np.pi*i/55)-1), 0., 0.)
    if kind == "ambiguity":
        size = np.exp(np.interp(s, [0, .25, .5, .75, 1], np.log([22, 45, 140, 45, 22])))
        return size, 4*i, 3*np.sin(2*np.pi*i/90), 4*np.sin(2*np.pi*i/115), 30*np.sin(2*np.pi*s), 0.
    if kind == "oblique":
        return (40*(1900/40)**s, 3*i, 45*np.sin(2*np.pi*i/140),
                25*np.sin(2*np.pi*i/115), 200*np.sin(2*np.pi*s), 120*np.sin(4*np.pi*s))
    if kind == "grazing":
        return (120*(1800/120)**s, 3*i, 55+20*np.sin(2*np.pi*i/160),
                20*np.sin(2*np.pi*i/125), 160*np.sin(2*np.pi*s), 100*np.sin(4*np.pi*s))
    if kind == "fast":
        # Finite differences of this integrated yaw are approximately 6..14 deg/frame.
        yaw = 10*i + 4*35/(2*np.pi)*(1-np.cos(2*np.pi*i/35))
        return (70*(1600/70)**s, yaw, 55*np.sin(2*np.pi*i/55),
                40*np.sin(2*np.pi*i/71), 180*np.sin(2*np.pi*s), 120*np.sin(4*np.pi*s))
    raise ValueError(kind)


def geometry(K, shape, real_size, points0, params, canvas, min_depth):
    height, width = shape[:2]
    sx, sy = real_size[0]/width, real_size[1]/height
    size, yaw, pitch, roll, dx, dy = params
    R = rotation_matrix(yaw, pitch, roll)
    corners = np.array([[-width/2,-height/2,0], [width/2,-height/2,0],
                        [width/2,height/2,0], [-width/2,height/2,0]]) * [sx,sy,1]
    requested_z = K[0,0]*min(real_size)/size
    z = max(requested_z, min_depth - (corners @ R.T)[:,2].min())
    center = np.array(canvas)/2 + [dx,dy]
    t = np.array([(center[0]-K[0,2])*z/K[0,0], (center[1]-K[1,2])*z/K[1,1], z])
    shift = np.array([-width/2*sx, -height/2*sy, 0])
    H = K @ np.column_stack([R[:,0]*sx, R[:,1]*sy, t+R@shift])
    p0 = np.asarray(points0)
    origin = (p0.min(axis=0)+p0.max(axis=0))/2
    origin_mm = np.r_[(origin-[width/2,height/2])*[sx,sy], 0.]
    t0 = t + R@origin_mm
    return H, R, t0, {
        "requested_size_px": float(size), "effective_size_px": float(K[0,0]*min(real_size)/z),
        "depth_capped": bool(z > requested_z+1e-8),
        "min_board_depth_mm": float(z+(corners@R.T)[:,2].min()),
        "yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": float(roll),
        "tilt_deg": float(np.rad2deg(np.arccos(np.clip(R[2,2],-1,1))))}


def generate_suite(config_path, out):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    tracker = json.loads(Path(cfg["tracker_config"]).read_text(encoding="utf-8-sig"))
    out = Path(out)
    # Refuse mixing old and new runs; never silently overwrite prior experiment frames.
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    target, bg = cv2.imread(cfg["target"]), cv2.imread(cfg["background"])
    if target is None or bg is None:
        raise ValueError("Missing target/background")
    if cfg.get("background_clean_rect"):
        # The original background file is absent. Recover it from the old first
        # frame, removing its 20px target before generating any new images.
        x,y,w,h = cfg["background_clean_rect"]
        mask = np.zeros(bg.shape[:2],np.uint8)
        mask[y:y+h,x:x+w] = 255
        bg = cv2.inpaint(bg,mask,5,cv2.INPAINT_TELEA)
    canvas = tuple(cfg["canvas"])
    bg = cv2.resize(bg, canvas, interpolation=cv2.INTER_AREA)
    points = [load_points(cfg[f"class{j}_points"]) for j in (0,1)]
    camera = tracker["camera"]
    K = np.array([[camera["fx"],0,camera["cx"]],[0,camera["fy"],camera["cy"]],[0,0,1.]])
    board = tracker["strategies"]["board_template"]
    real = [board["template_real_width_mm"],board["template_real_height_mm"]]
    height,width = target.shape[:2]
    centers = [(np.min(p,axis=0)+np.max(p,axis=0))/2 for p in points]
    actual_offset = (centers[1]-centers[0])*[real[0]/width,real[1]/height]
    suite = {"version":2, "config":cfg, "K":K.tolist(), "target_size_px":[width,height],
             "board_size_mm":real, "class1_actual_offset_mm":actual_offset.tolist(),
             "tracker_config_sha256":digest(cfg["tracker_config"]),
             "sources_sha256":{cfg[k]:digest(cfg[k]) for k in ("target","background","class0_points","class1_points")},
             "rotation_convention":"Rz(yaw) @ Rx(pitch) @ Ry(roll); target-to-camera",
             "origin":"class0 annotation bbox center", "profiles":[]}
    for profile in cfg["profiles"]:
        folder = out/profile["name"]
        folder.mkdir()
        rng = np.random.default_rng(cfg["seed"])
        manifest = {k:v for k,v in suite.items() if k != "profiles"}
        manifest.update(profile=profile, frames=[])
        for i in range(profile["frames"]):
            params = trajectory(profile["trajectory"],i,profile["frames"])
            H,R,t,meta = geometry(K,target.shape,real,points[0],params,canvas,cfg["minimum_board_depth_mm"])
            image = render(bg,target,H)
            degradation = {}
            occluder = None
            if profile.get("degradation"):
                sigma = .7+.7*(1+np.sin(i*.11))/2
                contrast = .65+.3*(1+np.cos(i*.07))/2
                image = cv2.GaussianBlur(image,(5,5),sigma)
                # Monochrome sensor noise, deterministic per frame; preserve color channels.
                noise = rng.normal(0,6,(*image.shape[:2],1)).astype(np.float32)
                image = np.clip((image.astype(np.float32)-127.5)*contrast+127.5+noise,0,255).astype(np.uint8)
                if 105 <= i < 125:
                    p = np.asarray(project_points(H,points[0]))
                    lo,hi = p.min(axis=0),p.max(axis=0)
                    x1,y1 = np.maximum(0,lo).astype(int)
                    x2,y2 = np.minimum(np.array(canvas)-1,[lo[0]+.25*(hi[0]-lo[0]),hi[1]]).astype(int)
                    if x2>x1 and y2>y1:
                        occluder = [int(x1),int(y1),int(x2),int(y2)]
                        cv2.rectangle(image,(x1,y1),(x2,y2),(90,90,90),-1)
                degradation = dict(blur_sigma=float(sigma),contrast=float(contrast),noise_sigma=6,occluder=occluder)
            path = f"frame_{i:06d}.png"
            if not cv2.imwrite(str(folder/path),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise IOError(folder/path)
            entry = dict(frame=i,path=path,R=R.tolist(),t_mm=t.tolist(),normal=R[:,2].tolist(),H=H.tolist(),
                         degradation=degradation,**meta)
            for j in (0,1):
                p = np.asarray(project_points(H,points[j]))
                inside = (p[:,0]>=0)&(p[:,0]<canvas[0])&(p[:,1]>=0)&(p[:,1]<canvas[1])
                visible = inside.copy()
                if occluder:
                    x1,y1,x2,y2 = occluder
                    visible &= ~((p[:,0]>=x1)&(p[:,0]<=x2)&(p[:,1]>=y1)&(p[:,1]<=y2))
                entry[f"class{j}"] = dict(points_px=p.tolist(),inside=inside.tolist(),visible=visible.tolist())
                write_points(str(folder/f"frame_{i:06d}_class{j}.txt"),p[visible],f"Class {j}",canvas)
            manifest["frames"].append(entry)
            if i%30==0:
                print(f"{profile['name']}: {i}/{profile['frames']} tilt={meta['tilt_deg']:.1f} size={meta['effective_size_px']:.1f}",flush=True)
        (folder/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        suite["profiles"].append(dict(name=profile["name"],frames=profile["frames"],manifest=f"{profile['name']}/manifest.json"))
    (out/"suite_manifest.json").write_text(json.dumps(suite,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Generated {sum(x['frames'] for x in suite['profiles'])} frames at {out}",flush=True)
    return 0
