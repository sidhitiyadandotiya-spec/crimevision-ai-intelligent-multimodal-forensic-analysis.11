"""
CrimeVision AI - 3D Crime Scene Spatial Reconstruction Engine
Provides 3D spatial coordinate data, architectural boundaries, evidence markers,
ballistic trajectory vectors, and camera angles for Three.js WebGL visualization.
"""
from typing import Dict, Any, List

class Scene3DEngine:
    def get_scene_3d_data(self, case_id: str) -> Dict[str, Any]:
        """
        Returns complete 3D scene data for interactive WebGL rendering.
        """
        if "0915" in case_id:
            return self._build_vault_3d_scene()
        return self._build_store_3d_scene()

    def _build_store_3d_scene(self) -> Dict[str, Any]:
        return {
            "scene_id": "SCENE-3D-0842",
            "name": "QuickStop Convenience Store - Crime Scene Spatial Model",
            "dimensions": {"width": 10.0, "length": 14.0, "height": 3.4},
            "room_elements": [
                {"type": "floor", "pos": [0, 0, 0], "size": [10, 0.1, 14], "color": "#1e293b", "label": "Polished Vinyl Tile Floor"},
                {"type": "wall", "pos": [0, 1.7, -7], "size": [10, 3.4, 0.2], "color": "#0f172a", "label": "North Wall (Entrance Glass & Doors)"},
                {"type": "wall", "pos": [0, 1.7, 7], "size": [10, 3.4, 0.2], "color": "#0f172a", "label": "South Wall (Backroom Shelving)"},
                {"type": "wall", "pos": [-5, 1.7, 0], "size": [0.2, 3.4, 14], "color": "#0f172a", "label": "West Wall (Cooler Units)"},
                {"type": "wall", "pos": [5, 1.7, 0], "size": [0.2, 3.4, 14], "color": "#0f172a", "label": "East Wall (Snack Displays)"},
                # Store Counter
                {"type": "counter", "pos": [0.6, 0.55, 0.8], "size": [2.6, 1.1, 1.0], "color": "#334155", "label": "Point-of-Sale Counter"},
                {"type": "register", "pos": [0.3, 1.15, 0.8], "size": [0.45, 0.35, 0.45], "color": "#475569", "label": "Touchscreen Till #1"},
                # Rear Merchandise Rack
                {"type": "shelf", "pos": [0.6, 1.4, 2.6], "size": [3.2, 2.6, 0.5], "color": "#1e293b", "label": "Cigarette & High-Value Inventory Rack"}
            ],
            "evidence_markers": [
                {
                    "marker_id": "MKR-01",
                    "label": "Evidence #1: Semi-Automatic Handgun Discharge Point",
                    "category": "weapon_discharge",
                    "pos": [-0.8, 1.25, -0.4],
                    "color": "#ef4444",
                    "icon": "crosshair",
                    "details": "Approximate 3D coordinates where 9mm pistol was leveled and discharged."
                },
                {
                    "marker_id": "MKR-02",
                    "label": "Evidence #2: Spent 9x19mm Brass Casing",
                    "category": "ballistics",
                    "pos": [-0.35, 0.05, -0.65],
                    "color": "#f59e0b",
                    "icon": "disc",
                    "details": "Recovered brass cartridge casing ejected to the right-rear of weapon position."
                },
                {
                    "marker_id": "MKR-03",
                    "label": "Evidence #3: Terminal Ballistic Impact Crater",
                    "category": "ballistics",
                    "pos": [1.1, 1.82, 2.58],
                    "color": "#dc2626",
                    "icon": "target",
                    "details": "Bullet impact in upper metal frame of cigarette rack shelf #3; slug embedded."
                },
                {
                    "marker_id": "MKR-04",
                    "label": "Evidence #4: Cash Drawer Contact Area",
                    "category": "latent_evidence",
                    "pos": [0.3, 1.15, 0.8],
                    "color": "#3b82f6",
                    "icon": "fingerprint",
                    "details": "Fingerprint and DNA swab location on register drawer rim."
                },
                {
                    "marker_id": "MKR-05",
                    "label": "Evidence #5: Primary Point of Ingress/Egress",
                    "category": "entry_exit",
                    "pos": [-3.5, 0.1, -6.8],
                    "color": "#10b981",
                    "icon": "log-in",
                    "details": "Double-door glass entrance with electronic contact sensor."
                }
            ],
            "suspect_trajectory": {
                "color": "#ef4444",
                "arrow_color": "#f87171",
                "waypoints": [
                    {"time": "00:01.16", "pos": [-3.5, 0.1, -6.8], "label": "1. Ingress"},
                    {"time": "00:03.66", "pos": [-1.8, 0.1, -2.5], "label": "2. Approach"},
                    {"time": "00:06.82", "pos": [-0.8, 0.1, -0.4], "label": "3. Discharge Weapon"},
                    {"time": "00:14.16", "pos": [-0.2, 0.1, 0.3], "label": "4. Cash Seizure"},
                    {"time": "00:19.37", "pos": [-3.5, 0.1, -6.8], "label": "5. Egress"}
                ]
            },
            "ballistic_trajectory": {
                "origin": [-0.8, 1.25, -0.4],
                "destination": [1.1, 1.82, 2.58],
                "color": "#f43f5e",
                "distance_meters": 3.52,
                "elevation_deg": 12.4,
                "notes": "Trajectory path from muzzle position to terminal shelf impact."
            },
            "camera_presets": [
                {
                    "id": "cctv_cam1",
                    "label": "Surveillance Camera 01 (Overhead Angle)",
                    "pos": [-3.8, 2.9, -5.8],
                    "look_at": [0.2, 0.8, 0.5]
                },
                {
                    "id": "clerk_pov",
                    "label": "Clerk Point of View (Behind Counter)",
                    "pos": [0.6, 1.6, 1.5],
                    "look_at": [-0.8, 1.3, -0.4]
                },
                {
                    "id": "suspect_pov",
                    "label": "Suspect Point of View (Entering Store)",
                    "pos": [-3.5, 1.7, -6.0],
                    "look_at": [0.6, 1.1, 0.8]
                },
                {
                    "id": "top_down",
                    "label": "2D Floorplan Top-Down View",
                    "pos": [0, 12.0, 0],
                    "look_at": [0, 0, 0]
                },
                {
                    "id": "free_orbit",
                    "label": "360° Free Orbit Mode",
                    "pos": [-5.0, 4.0, -4.0],
                    "look_at": [0, 1.0, 0]
                }
            ]
        }

    def _build_vault_3d_scene(self) -> Dict[str, Any]:
        return {
            "scene_id": "SCENE-3D-0915",
            "name": "Commercial Jewelry Vault - Crime Scene 3D Reconstruction",
            "dimensions": {"width": 8.0, "length": 8.0, "height": 3.0},
            "room_elements": [
                {"type": "floor", "pos": [0, 0, 0], "size": [8, 0.1, 8], "color": "#1e293b", "label": "Concrete Subfloor"},
                {"type": "safe", "pos": [-2.0, 0.9, -1.5], "size": [1.4, 1.8, 1.2], "color": "#475569", "label": "Commercial Safe #2 (Breached)"},
                {"type": "tool_bag", "pos": [0.2, 0.2, 0.5], "size": [0.6, 0.35, 0.4], "color": "#b45309", "label": "Discarded Tool Bag"}
            ],
            "evidence_markers": [
                {
                    "marker_id": "MKR-0915-01",
                    "label": "Evidence #1: Breached Safe Door Lock Hinge",
                    "category": "breach_point",
                    "pos": [-1.6, 1.0, -1.0],
                    "color": "#ef4444",
                    "icon": "shield-alert",
                    "details": "Pry marks and sheared locking bolts on safe perimeter."
                },
                {
                    "marker_id": "MKR-0915-02",
                    "label": "Evidence #2: Tool Bag & Dropped Lockpicks",
                    "category": "tools",
                    "pos": [0.2, 0.1, 0.5],
                    "color": "#f59e0b",
                    "icon": "briefcase",
                    "details": "Left at scene containing serial-less pry bar and tension wrenches."
                }
            ],
            "suspect_trajectory": {
                "color": "#f59e0b",
                "arrow_color": "#fbbf24",
                "waypoints": [
                    {"time": "03:22.00", "pos": [3.0, 0.1, 3.0], "label": "1. Entry Hatch"},
                    {"time": "03:22.41", "pos": [-1.2, 0.1, -0.8], "label": "2. Kneeling at Safe"},
                    {"time": "03:25.10", "pos": [3.0, 0.1, 3.0], "label": "3. Egress"}
                ]
            },
            "camera_presets": [
                {"id": "cam_vault", "label": "Night-Vision CAM-04 Angle", "pos": [3.2, 2.5, 3.2], "look_at": [-1.8, 0.9, -1.2]},
                {"id": "top_down", "label": "Overhead Floorplan", "pos": [0, 9.0, 0], "look_at": [0, 0, 0]}
            ]
        }

# Global singleton
scene_3d = Scene3DEngine()
