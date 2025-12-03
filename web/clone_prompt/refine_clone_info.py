import os

def refine_file(input_filename, output_filename):
    # 현재 실행 위치 확인 (디버깅용)
    print(f"작업 시작: {input_filename} 읽기 시도...")

    try:
        # 1. 파일 읽기 (utf-8 인코딩)
        with open(input_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        refined_lines = []
        
        for line in lines:
            # 2. 로직: strip()으로 확인했을 때 내용이 있는 줄만 남김
            # (공백이나 탭만 있는 줄은 line.strip()이 빈 문자열이 되어 False 처리됨)
            if line.strip():
                # 원본 라인의 오른쪽 공백(기존 줄바꿈 포함)만 제거하고 리스트에 추가
                # *중요*: 왼쪽 공백(들여쓰기)은 유지해야 HTML/CSS 구조가 깨지지 않음
                refined_lines.append(line.rstrip())

        # 3. 파일 쓰기
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 리스트에 있는 각 줄을 새로운 줄바꿈(\n)으로 연결하여 저장
            f.write('\n'.join(refined_lines))

        print(f"완료! 정제된 파일이 생성되었습니다: {output_filename}")
        print(f"제거된 빈 줄 수: {len(lines) - len(refined_lines)}줄")

    except FileNotFoundError:
        print(f"[오류] '{input_filename}' 파일을 찾을 수 없습니다.")
        print("파이썬 파일과 같은 폴더에 txt 파일이 있는지 확인해주세요.")
    except Exception as e:
        print(f"[오류] 알 수 없는 문제가 발생했습니다: {e}")

if __name__ == "__main__":
    # 입력 파일명과 출력 파일명 지정
    INPUT_FILE = 'clone_제안신청_info.txt'
    OUTPUT_FILE = 'clone_제안신청_info_refined.txt'
    
    refine_file(INPUT_FILE, OUTPUT_FILE)