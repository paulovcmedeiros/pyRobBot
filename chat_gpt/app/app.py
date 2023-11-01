import shutil

from st_pages import Page, add_page_title, show_pages

from chat_gpt import GeneralConstants

# Optional -- adds the title and icon to the current page
add_page_title()

pkg_root = GeneralConstants.PACKAGE_DIRECTORY / "app"
n_pages = 2

pages = []
for ipage in range(n_pages):
    page_path = GeneralConstants.PACKAGE_TMPDIR / f"app_page_{ipage+1}.py"
    shutil.copy(src=pkg_root / "pages/template.py", dst=page_path)
    pages.append(Page(page_path.as_posix(), f"Chat {ipage+1}", ":books:"))


# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be
show_pages(pages)
