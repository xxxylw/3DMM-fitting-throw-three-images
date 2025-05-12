def save_obj(path, verts, faces, colors):
    """
    保存带顶点颜色的 OBJ 文件。
    参数:
      path   - 输出 OBJ 路径
      verts  - 顶点列表，shape [V,3]
      faces  - 面列表，shape [F,3]，1-based 索引
      colors - 顶点颜色，shape [V,3], 值域 [0,1] 或 [0,255]
    """
    with open(path, 'w') as f:
        f.write('# OBJ with per-vertex color\n')
        for i, v in enumerate(verts):
            # 如果 colors 范围在 [0,1]，转到 [0,255]
            c = colors[i]
            if c.max() <= 1.0:
                c = (c * 255).astype(np.uint8)
            f.write('v %f %f %f %d %d %d\n' % (
                v[0], v[1], v[2], int(c[0]), int(c[1]), int(c[2])
            ))
        for face in faces:
            # faces 已经是 1-based 索引
            f.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print(f"OBJ with vertex colors saved to {path}")