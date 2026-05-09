find_package(Doxygen REQUIRED)

set(YORCH_DOXYGEN_OUTPUT_DIR "${PROJECT_BINARY_DIR}/target")
set(YORCH_DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/input")
set(YORCH_DOXYGEN_WIKI_EN_DIR "${YORCH_DOXYGEN_INPUT_DIR}/wiki/en")
set(YORCH_DOXYGEN_WIKI_CN_DIR "${YORCH_DOXYGEN_INPUT_DIR}/wiki/cn")
set(YORCH_DOXYGEN_MAINPAGE "${YORCH_DOXYGEN_INPUT_DIR}/index.md")
set(YORCH_DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(YORCH_DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

file(MAKE_DIRECTORY "${YORCH_DOXYGEN_INPUT_DIR}")
file(MAKE_DIRECTORY "${YORCH_DOXYGEN_WIKI_EN_DIR}")
file(MAKE_DIRECTORY "${YORCH_DOXYGEN_WIKI_CN_DIR}")

file(READ "${PROJECT_SOURCE_DIR}/README.md" YORCH_DOXYGEN_README)
string(REPLACE "src=\"./assets/banner.png\"" "src=\"assets/banner.png\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
string(REPLACE "href=\"LICENSE\"" "href=\"${YORCH_REPOSITORY_URL}/blob/HEAD/LICENSE\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
string(REPLACE "href=\"README.md\"" "href=\"index.html\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
string(REPLACE "href=\"wiki/cn/intro_cn.md\"" "href=\"md_intro__cn.html\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
string(REPLACE "href=\"wiki/en/register_task.md\"" "href=\"md_wiki_2en_2register__task.html\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
string(REPLACE "href=\"wiki/en/build_tree.md\"" "href=\"md_wiki_2en_2build__tree.html\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
string(REPLACE "href=\"wiki/en/executor.md\"" "href=\"md_wiki_2en_2executor.html\"" YORCH_DOXYGEN_README "${YORCH_DOXYGEN_README}")
file(WRITE "${YORCH_DOXYGEN_MAINPAGE}" "${YORCH_DOXYGEN_README}")

file(READ "${PROJECT_SOURCE_DIR}/wiki/cn/intro_cn.md" YORCH_DOXYGEN_INTRO_CN)
string(REPLACE "src=\"../../assets/banner.png\"" "src=\"assets/banner.png\"" YORCH_DOXYGEN_INTRO_CN "${YORCH_DOXYGEN_INTRO_CN}")
string(REPLACE "href=\"../../LICENSE\"" "href=\"${YORCH_REPOSITORY_URL}/blob/HEAD/LICENSE\"" YORCH_DOXYGEN_INTRO_CN "${YORCH_DOXYGEN_INTRO_CN}")
string(REPLACE "href=\"../../README.md\"" "href=\"index.html\"" YORCH_DOXYGEN_INTRO_CN "${YORCH_DOXYGEN_INTRO_CN}")
string(REPLACE "href=\"intro_cn.md\"" "href=\"md_intro__cn.html\"" YORCH_DOXYGEN_INTRO_CN "${YORCH_DOXYGEN_INTRO_CN}")
file(WRITE "${YORCH_DOXYGEN_INPUT_DIR}/intro_cn.md" "${YORCH_DOXYGEN_INTRO_CN}")

foreach(YORCH_WIKI_PAGE IN ITEMS build_tree executor register_task)
  string(REPLACE "_" "__" YORCH_WIKI_HTML_STEM "${YORCH_WIKI_PAGE}")

  file(READ "${PROJECT_SOURCE_DIR}/wiki/en/${YORCH_WIKI_PAGE}.md" YORCH_DOXYGEN_WIKI_EN)
  string(REPLACE "href=\"${YORCH_WIKI_PAGE}.md\"" "href=\"md_wiki_2en_2${YORCH_WIKI_HTML_STEM}.html\"" YORCH_DOXYGEN_WIKI_EN "${YORCH_DOXYGEN_WIKI_EN}")
  string(REPLACE "href=\"../cn/${YORCH_WIKI_PAGE}.md\"" "href=\"md_wiki_2cn_2${YORCH_WIKI_HTML_STEM}.html\"" YORCH_DOXYGEN_WIKI_EN "${YORCH_DOXYGEN_WIKI_EN}")
  file(WRITE "${YORCH_DOXYGEN_WIKI_EN_DIR}/${YORCH_WIKI_PAGE}.md" "${YORCH_DOXYGEN_WIKI_EN}")

  file(READ "${PROJECT_SOURCE_DIR}/wiki/cn/${YORCH_WIKI_PAGE}.md" YORCH_DOXYGEN_WIKI_CN)
  string(REPLACE "[English](../en/${YORCH_WIKI_PAGE}.md) | [中文](${YORCH_WIKI_PAGE}.md)" "<a href=\"md_wiki_2en_2${YORCH_WIKI_HTML_STEM}.html\">English</a> | <a href=\"md_wiki_2cn_2${YORCH_WIKI_HTML_STEM}.html\">中文</a>" YORCH_DOXYGEN_WIKI_CN "${YORCH_DOXYGEN_WIKI_CN}")
  file(WRITE "${YORCH_DOXYGEN_WIKI_CN_DIR}/${YORCH_WIKI_PAGE}.md" "${YORCH_DOXYGEN_WIKI_CN}")
endforeach()

configure_file("${YORCH_DOXYFILE_IN}" "${YORCH_DOXYFILE_OUT}" @ONLY)

add_custom_target(docs
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${YORCH_DOXYGEN_OUTPUT_DIR}/html"
  COMMAND "${DOXYGEN_EXECUTABLE}" "${YORCH_DOXYFILE_OUT}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${PROJECT_SOURCE_DIR}/assets"
    "${YORCH_DOXYGEN_OUTPUT_DIR}/html/assets"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM
)
