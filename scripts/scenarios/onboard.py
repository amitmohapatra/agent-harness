"""Onboard the scen tenant (and a second tenant, for the tenant-boundary scenarios) with the
bootstrap key, issue service keys; append them to keys.env (never printed)."""

import asyncio
import os
import sys
from pathlib import Path

from trellis.memory import MemoryClient

URL = f"http://127.0.0.1:{os.environ['SCEN_MEM_PORT']}"
TENANT = os.environ["SCEN_TENANT"]


async def onboard(tenant: str, name: str) -> tuple[str, str]:
    async with MemoryClient(URL, api_key=os.environ["SCEN_BOOTSTRAP_KEY"]) as platform:
        created = await platform.admin.create_tenant(name, tenant_id=tenant)
        admin_key = created.admin_key.token
    async with MemoryClient(URL, api_key=admin_key) as admin:
        issued = await admin.administer(tenant).keys.issue("service", f"{tenant}-harness")
    return admin_key, issued.token


async def main() -> None:
    admin, key = await onboard(TENANT, "Scenarios")
    admin2, key2 = await onboard(TENANT + "2", "Scenarios Other")
    with (Path(os.environ["SCEN_HOME"]) / "keys.env").open("a") as out:
        out.write(
            f"export SCEN_ADMIN_KEY={admin}\nexport SCEN_KEY={key}\n"
            f"export SCEN_ADMIN_KEY2={admin2}\nexport SCEN_KEY2={key2}\n"
        )
    sys.stdout.write(f"tenants onboarded: {TENANT}, {TENANT}2\n")


asyncio.run(main())
