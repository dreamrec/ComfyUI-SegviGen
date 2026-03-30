/**
 * SegviGen Load Mesh Widget
 *
 * Adds a "📁 Upload Mesh File" button to the SegviGenLoadMesh node so users
 * can browse their filesystem directly instead of typing a path.
 *
 * On file selection the mesh is uploaded to ComfyUI/input/3d/ via the
 * standard /upload/image endpoint (ComfyUI accepts any file there when
 * given a subfolder), then the model_file combo widget is updated to point
 * at the newly uploaded file.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SegviGen.LoadMesh",

    nodeCreated(node) {
        if (node.comfyClass !== "SegviGenLoadMesh") return;

        // Add upload button right after the existing model_file combo.
        node.addWidget(
            "button",
            "📁 Upload Mesh File",
            null,
            async () => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = ".glb,.gltf,.obj,.ply,.fbx,.stl";

                input.onchange = async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;

                    // Upload to input/3d/ using ComfyUI's built-in endpoint.
                    const formData = new FormData();
                    formData.append("image", file, file.name);
                    formData.append("subfolder", "3d");
                    formData.append("type", "input");
                    formData.append("overwrite", "true");

                    let data;
                    try {
                        const resp = await fetch("/upload/image", {
                            method: "POST",
                            body: formData,
                        });
                        if (!resp.ok) {
                            throw new Error(`Upload failed: ${resp.statusText}`);
                        }
                        data = await resp.json();
                    } catch (err) {
                        alert(`SegviGen: mesh upload failed.\n\n${err.message}`);
                        return;
                    }

                    // ComfyUI returns { name, subfolder, type }.
                    // The path the node expects is "3d/<filename>".
                    const uploadedPath = data.subfolder
                        ? `${data.subfolder}/${data.name}`
                        : data.name;

                    // Update the model_file combo widget.
                    const combo = node.widgets?.find(
                        (w) => w.name === "model_file"
                    );
                    if (combo) {
                        // Dynamically add the path to the options if absent.
                        if (
                            combo.options?.values &&
                            !combo.options.values.includes(uploadedPath)
                        ) {
                            combo.options.values = [
                                uploadedPath,
                                ...combo.options.values,
                            ];
                        }
                        combo.value = uploadedPath;
                        app.graph.setDirtyCanvas(true);
                        console.log(
                            `[SegviGen] uploaded mesh → ${uploadedPath}`
                        );
                    }
                };

                input.click();
            }
        );
    },
});
