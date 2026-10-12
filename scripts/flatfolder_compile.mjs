// Headless Flat-Folder compile: node scripts/flatfolder_compile.mjs [--limit N] <manifest.json>
//
// manifest.json = {"inputs": ["a.fold", ...], "output": "results.json"}
// For each input FOLD file, replicates Flat-Folder's batch pipeline (src/batch.js) and
// writes one record per input in the schema of PurelandFold's `flat_folder` column:
//   compilation_success, reason, Vf, Ff, faceOrders, taco_taco, taco_tortilla,
//   tortilla_tortilla, transitivity, reduced_transitivity, num_folded_states,
//   num_components, counts{vertices, edges, faces, points, segments, cells, variables, eps}
// faceOrders / Ff are expressed in the INPUT file's face indices (Flat-Folder sorts faces
// by area internally; we undo that permutation).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const FF_SRC = process.env.FLATFOLDER_SRC ?? path.join(here, "..", "external", "flat-folder", "src");
const load = async (name) => import(pathToFileURL(path.join(FF_SRC, name)).href);
const { M } = await load("math.js");
const { NOTE } = await load("note.js");
const { CON } = await load("constraints.js");
const { IO } = await load("io.js");
const { X } = await load("conversion.js");
const { SOLVER } = await load("solver.js");

for (const k of ["log", "time", "count", "annotate", "lap", "start", "end", "clear_log"]) {
    NOTE[k] = () => {};
}
NOTE.show = false;
CON.build();

const faceKey = (F) => [...F].sort((a, b) => a - b).join(",");

function compile(doc, lim) {
    const original = JSON.parse(doc);
    const origFV = (original.faces_vertices ?? []).map(faceKey);
    const [V, , EV, EA, EF, FV, FE] = IO.doc_type_side_2_V_VV_EV_EA_EF_FV_FE(doc, "fold", true);
    if (V == undefined) { return { compilation_success: false, reason: "import failed" }; }
    // Flat-Folder face index -> input face index.
    const toOrig = FV.map((F) => origFV.indexOf(faceKey(F)));
    const [Vf, Ff] = X.V_FV_EV_EA_2_Vf_Ff(V, FV, EV, EA);
    const L = EV.map((P) => M.expand(P, Vf));
    const [P, SP, SE, eps_i] = X.L_2_V_EV_EL(L);
    if (P.length == 0) {
        return { compilation_success: false, reason: "precision error: could not find stable graph" };
    }
    const [, CP] = X.V_EV_2_VV_FV(P, SP);
    const [SC] = X.EV_FV_2_EF_FE(SP, CP);
    const [CF, FC] = X.EF_FV_P_SP_SE_CP_SC_2_CF_FC(EF, FV, P, SP, SE, CP, SC);
    const BF = X.EF_SP_SE_CP_CF_2_BF(EF, SP, SE, CP, CF);
    const BI = new Map();
    for (const [i, F] of BF.entries()) { BI.set(F, i); }
    const BT = X.BF_BI_EF_SE_CF_SC_2_BT(BF, BI, EF, SE, CF, SC);
    const BTn = [0, 0, 0];
    for (const bT of BT) { for (let i = 0; i < 3; ++i) { BTn[i] += bT[i].length; } }
    for (const [i, d] of [[0, 6], [1, 2], [2, 2]]) { BTn[i] /= d; }
    const CC = X.FC_BF_BI_BT_2_CC(FC, BF, BI, BT);
    const BA0 = SOLVER.EF_EA_Ff_BF_BI_2_BA0(EF, EA, Ff, BF, BI);
    const trans_count = { all: 0, reduced: 0 };
    const counts = {
        vertices: V.length, edges: EV.length, faces: FV.length, points: P.length,
        segments: SP.length, cells: CP.length, variables: BF.length, eps: eps_i,
    };
    const partial = {
        Vf, Ff: toOrig.map((_, i) => Ff[i]),
        taco_taco: BTn[0], taco_tortilla: BTn[1], tortilla_tortilla: BTn[2], counts,
    };
    const out = SOLVER.initial_assignment(BA0, BF, BT, BI, FC, CF, CC, trans_count);
    if (out.length == 3 && Array.isArray(out[1])) {
        const [type, F] = out;
        const faces = F.map((f) => toOrig[f]);
        return {
            compilation_success: false,
            reason: `unable to resolve ${CON.names[type]} on faces [${faces}]`,
            ...partial, transitivity: trans_count.all / 3,
            reduced_transitivity: trans_count.reduced / 3,
        };
    }
    const BA = out;
    const GB = SOLVER.get_components(BI, BF, BT, BA, FC, CF, CC, trans_count);
    const GA = SOLVER.solve(BI, BF, BT, BA.map((a) => a), GB, FC, CF, CC, lim);
    if (GA.length == undefined) {
        return {
            compilation_success: false, reason: `unable to resolve component ${GA}`,
            ...partial, transitivity: trans_count.all / 3,
            reduced_transitivity: trans_count.reduced / 3, num_components: GB.length,
        };
    }
    const n = GA.reduce((s, A) => s * BigInt(A.length), 1n);
    const Gi = GA.map(() => 0);
    const edges = X.BF_GB_GA_GI_2_edges(BF, GB, GA, Gi);
    const FO = X.edges_Ff_2_FO(edges, Ff).map(([a, b, s]) => [toOrig[a], toOrig[b], s]);
    // Undo the face permutation on Ff as well.
    const FfOrig = new Array(origFV.length).fill(false);
    for (const [i, o] of toOrig.entries()) { if (o >= 0) { FfOrig[o] = Ff[i]; } }
    return {
        compilation_success: true, reason: "",
        Vf, Ff: FfOrig, faceOrders: FO,
        taco_taco: BTn[0], taco_tortilla: BTn[1], tortilla_tortilla: BTn[2],
        transitivity: trans_count.all / 3, reduced_transitivity: trans_count.reduced / 3,
        num_folded_states: n.toString(), num_components: GB.length, counts,
    };
}

const args = process.argv.slice(2);
let lim = 1000;
const li = args.indexOf("--limit");
if (li >= 0) { lim = args[li + 1] == "all" ? Infinity : +args[li + 1]; args.splice(li, 2); }
const manifest = JSON.parse(fs.readFileSync(args[0], "utf8"));
const results = [];
for (const file of manifest.inputs) {
    const t0 = performance.now();
    let rec;
    try {
        rec = compile(fs.readFileSync(file, "utf8"), lim);
    } catch (e) {
        rec = { compilation_success: false, reason: `exception: ${e?.message ?? e}` };
    }
    rec.input = file;
    rec.seconds = Math.round(performance.now() - t0) / 1000;
    results.push(rec);
}
fs.writeFileSync(manifest.output, JSON.stringify(results));
