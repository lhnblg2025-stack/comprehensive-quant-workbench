# ⚠️ 已废弃（2026-08-22 标注）：本文件是 Windows(C:\quant) 时代遗留的日报启动器，
# 在 Linux 部署下 sys.path.insert(r"C:\quant") 与 C:\quant\.task_lock 必失败，属死脚本。
# 正式日报入口: scripts/a_share_daily_report.py（直接运行）; 复盘链入口: scripts/daily_review_chain.py。
# 本文件保留仅供历史审计，勿再用于调度。
import logging
import sys, os, time, traceback
sys.path.insert(0, r"C:\quant")
try:
    # 单栈化(V3)：改走 quant_platform.legacy 薄转发，不再直接依赖 quant_v6
    from quant_platform.legacy import nice_process
    nice_process()
except Exception as e:
    logging.getLogger(__name__).error(f"[daily_report_runner] 操作失败: {e}", exc_info=True)
LOCK = r"C:\quant\.task_lock"
def acquire_lock():
    for _ in range(60):
        if not os.path.exists(LOCK):
            try:
                open(LOCK, "w").close()
                return True
            except Exception as e:
                logging.getLogger(__name__).error(f"[daily_report_runner] 操作失败: {e}", exc_info=True)
        time.sleep(5)
    print("LOCK_TIMEOUT: another task running")
    return False
def release_lock():
    try:
        if os.path.exists(LOCK):
            os.remove(LOCK)
    except Exception as e:
        logging.getLogger(__name__).error(f"[daily_report_runner] 操作失败: {e}", exc_info=True)
if __name__ == "__main__":
    if not acquire_lock():
        sys.exit(2)
    rc = 0
    try:
        sys.path.insert(0, r"C:\quant\scripts")
        import argparse
        from a_share_daily_report import main as daily_main
        ap = argparse.ArgumentParser()
        ap.add_argument("--date", default=None)
        ap.add_argument("--out", default=None)
        ap.add_argument("--no-finalize", action="store_true")
        args = ap.parse_args()
        # a_share_daily_report.main() 无参，内部用 argparse 读 sys.argv → 注入
        sys.argv = ["daily_report_runner.py"]
        if args.date:
            sys.argv.append(f"--date={args.date}")
        if args.out:
            sys.argv.append(f"--out={args.out}")
        if args.no_finalize:
            sys.argv.append("--no-finalize")
        rc = int(daily_main() or 0)
        print("DAILY_DONE rc=", rc)
    except SystemExit as se:
        rc = int(se.code or 0)
        print("DAILY_EXIT rc=", rc)
    except Exception:
        rc = 3
        traceback.print_exc()
        with open(r"C:\quant\logs\daily_fail.log", "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc() + "\n")
    finally:
        release_lock()
    sys.exit(rc)
