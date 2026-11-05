add_executable(parakeet-server
    main.cpp
    model_fetch.cpp
    openai_format.cpp)
target_link_libraries(parakeet-server PRIVATE parakeet)
target_include_directories(parakeet-server PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party)
find_package(Threads REQUIRED)
target_link_libraries(parakeet-server PRIVATE Threads::Threads)
# Packaged builds set a system-wide model directory; empty keeps the per-user default.
set(PARAKEET_MODEL_DIR "" CACHE STRING "Default directory for fetched models")
if(PARAKEET_MODEL_DIR)
    target_compile_definitions(parakeet-server PRIVATE PARAKEET_MODEL_DIR="${PARAKEET_MODEL_DIR}")
endif()
