from .facegate_comfyui import FaceGateNode, FaceGateSavePassedImages


NODE_CLASS_MAPPINGS = {
    "FaceGate": FaceGateNode,
    "FaceGateSavePassedImages": FaceGateSavePassedImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceGate": "FaceGate Filter",
    "FaceGateSavePassedImages": "Facegate Save Image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
