"""Фронтальный каркасный рендер Spectrum Elite с обеими ветвями проекции."""

from .types import EdgeDecision, GeometryTrace, ProjectedVertex
from .arithmetic import (make_basis, rotate_components, translate_vertex_axis,
                         perspective_divide, line_of_sight_component,
                         face_dot_sign)


def _within_distance(limit: int, distance: int) -> bool:
    return limit == 31 or distance < limit


def classify_faces(model, pose, basis, distance, *, use_face_vertices=None):
    """$BCD4: флаг означает результат проверки нормали, а не отсутствие окклюзии."""
    flags = [1] * 16  # В том числе специальный индекс 15, как в оригинале.
    if use_face_vertices is None:
        use_face_vertices = model.vertex_test_default
    normals = []
    for index, face in enumerate(model.faces):
        rotated = rotate_components((face.nx, face.ny, face.nz), basis)
        normals.append(tuple(component.signed for component in rotated))
        if not _within_distance(face.distance_limit, distance):
            flags[index] = 0
            continue
        anchor = (None, None, None)
        if use_face_vertices and model.face_vertices:
            vertex = model.vertices[model.face_vertices[index]]
            anchor = rotate_components((vertex.x, vertex.y, vertex.z), basis)
        pairs = [line_of_sight_component(position, component, model.normal_scale, point)
                 for position, component, point in zip(pose.position, rotated, anchor)]
        normal, sight = zip(*pairs)
        flags[index] = int(face_dot_sign(normal, sight))
    return flags, normals


def projection_shift(vertex, distance: int) -> int:
    """$ACA4–$ACC1: число свободных старших битов в модулях одной вершины."""
    if distance >= 4:
        return 0
    magnitudes = abs(vertex.x) | abs(vertex.y) | abs(vertex.z)
    if magnitudes == 0:
        # У исходного цикла $ACB7 в этом случае нет выхода.
        raise ValueError("В ближней ветви Elite вершина в центре модели недопустима")
    return 8 - magnitudes.bit_length()


def project_vertices(model, pose, basis, distance, face_flags, *, precision_trace=None):
    """$A9C9 → $ACA2: отбор, выбор точности, поворот и перспектива."""
    projected = []
    previous_magnitudes = None
    transformed = []
    for index, vertex in enumerate(model.vertices):
        if (not _within_distance(vertex.distance_limit, distance)
                or not any(face_flags[face] for face in vertex.faces)):
            projected.append(None)
            transformed.append(None)
            previous_magnitudes = None  # $AA08: сброс кэша.
            continue
        vector = (vertex.x, vertex.y, vertex.z)
        magnitudes = tuple(map(abs, vector))
        shift = projection_shift(vertex, distance)
        # $AB1E сдвигает модули до табличного умножения. У соседних вершин
        # с одинаковыми модулями совпадает и shift, сохраняем порядок $AA49.
        scaled = tuple(component << shift for component in vector)
        rotated = rotate_components(scaled, basis, cached=magnitudes == previous_magnitudes)
        previous_magnitudes = magnitudes
        axes = []
        fractions = []
        carry = 0  # $ACDF: XOR A перед вычислением X.
        for position, component in zip(pose.position, rotated):
            axis, fraction, carry = translate_vertex_axis(position, component, shift, carry)
            axes.append(axis)
            fractions.append(fraction)
        x, y, z = axes
        xf, yf, zf = fractions
        if precision_trace is not None:
            precision_trace[index] = (shift, (xf, yf, zf))
        transformed.append((x.signed, y.signed, z.signed))
        xmag = perspective_divide(x.magnitude, z.magnitude, xf, zf)
        ymag = perspective_divide(y.magnitude, z.magnitude, yf, zf)
        flags = int(y.negative) | (int(x.negative) << 1)
        if z.negative:
            flags = 0xFF
        projected.append(ProjectedVertex(xmag, ymag, flags))
    return projected, transformed


def select_edges(model, distance, face_flags, projected):
    """$AE01: последовательность проверок ребра перед единственным CALL линии."""
    decisions = []
    for index, edge in enumerate(model.edges):
        if not _within_distance(edge.distance_limit, distance):
            decisions.append(EdgeDecision(index, False, "distance"))
            continue
        face_a, face_b = edge.faces
        if not (face_flags[face_a] or face_flags[face_b]):
            decisions.append(EdgeDecision(index, False, "faces"))
            continue
        start = projected[edge.vertex_a]
        end = projected[edge.vertex_b]
        if start is None or end is None:
            decisions.append(EdgeDecision(index, False, "vertex_not_projected", start, end))
            continue
        if start.invalid or end.invalid:
            decisions.append(EdgeDecision(index, False, "behind_camera", start, end))
            continue
        decisions.append(EdgeDecision(index, True, "draw", start, end))
    return decisions


def render_geometry(model, pose, *, use_face_vertices=None):
    """От исходных таблиц до отрезков; библиотека рисования здесь не участвует."""
    x, y, z = pose.position
    if x != 0 or y != 0 or not 1 <= z <= 6000:
        raise ValueError("Учебная версия: position=(0,0,z), 1 <= z <= 6000")
    if any(abs(value) > 127 for value in (*pose.nose, *pose.roof)):
        raise ValueError("Компоненты nose/roof должны лежать в [-127,127]")
    distance = max(0, min(31, z >> 7) - 2)  # $A854–$A867
    basis = make_basis(pose.nose, pose.roof)
    flags, normals = classify_faces(model, pose, basis, distance, use_face_vertices=use_face_vertices)
    precision = {}
    projected, transformed = project_vertices(model, pose, basis, distance, flags, precision_trace=precision)
    decisions = select_edges(model, distance, flags, projected)
    return GeometryTrace(distance, flags, projected, decisions,
                         details={"basis": [[component.signed for component in row]
                                            for row in basis],
                                  "normals_half_scale": normals,
                                  "transformed": transformed, "vertex_precision": precision})
