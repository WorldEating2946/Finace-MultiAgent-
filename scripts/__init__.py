"""一次性运维脚本包（init_db / verify_* / migrate_* 等）。

做成包使 ``from scripts.migrate_faiss_to_milvus import ...`` 可被 pytest 导入，
方便对迁移/校验核心函数做离线单测。
"""
