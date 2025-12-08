import React from 'react';
import styled from 'styled-components';

const TableWrapper = styled.div`
  width: 100%;
  border-top: 2px solid var(--gray80);
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;

  th, td {
    padding: 15px 10px;
    text-align: center;
    border-bottom: 1px solid var(--gray20);
    font-size: 1.5rem;
  }

  th {
    background-color: var(--gray5);
    font-weight: 600;
    color: var(--gray90);
  }

  td {
    color: var(--gray70);
  }

  td.title {
    text-align: left;
    padding-left: 20px;
    
    a {
      color: var(--gray90);
      &:hover {
        text-decoration: underline;
        color: var(--primary50);
      }
    }
  }
`;

const Badge = styled.span`
  display: inline-block;
  margin-left: 5px;
  padding: 2px 5px;
  background-color: var(--primary10);
  color: var(--primary60);
  font-size: 1.2rem;
  border-radius: 4px;
`;

const DataTable = () => {
    const data = [
        { id: 4026, title: "건강일터 조성사업, 온열질환 예방장비 지원 사업 관련", dept: "고용노동부", date: "2025-12-02", status: "제안심사", views: 8, badge: "공동제안" },
        { id: 4025, title: "AGI 관련 추가 백서입니다.", dept: "한국지능정보사회진흥원", date: "2025-12-02", status: "제안심사", views: 1 },
        { id: 4024, title: "사서교사 정원 확대 및 기간제 교사 배치 의무화 촉구", dept: "경기도교육청", date: "2025-12-02", status: "제안심사", views: 7 },
        { id: 4023, title: "등외 상이군인 보훈 사각지대 해소를 위한 국가유공자 8급 신설", dept: "국가보훈부", date: "2025-12-02", status: "제안심사", views: 8 },
        { id: 4022, title: "경로당 비상벨 설치 필요합니다.", dept: "보건복지부", date: "2025-12-02", status: "제안심사", views: 6 },
        { id: 4021, title: "사직운동장 풋살장 건립", dept: "부산광역시", date: "2025-12-02", status: "제안심사", views: 1 },
        { id: 4020, title: "등외판정 상이군인을 위한 국가유공자 8급 신설 제안", dept: "국가보훈부", date: "2025-12-02", status: "제안심사", views: 9 },
        { id: 4019, title: "국기봉 및 게양기의 개선 안.", dept: "행정안전부", date: "2025-12-02", status: "제안심사", views: 6 },
        { id: 4018, title: "국기 게양대 개선안", dept: "행정안전부", date: "2025-12-02", status: "제안심사", views: 6 },
        { id: 4017, title: "편의점 미성년자 담배구매 처벌 기준변경", dept: "성평등가족부", date: "2025-12-02", status: "제안심사", views: 8 },
    ];

    return (
        <TableWrapper>
            <Table>
                <colgroup>
                    <col style={{ width: '70px' }} />
                    <col style={{ width: 'auto' }} />
                    <col style={{ width: '150px' }} />
                    <col style={{ width: '110px' }} />
                    <col style={{ width: '100px' }} />
                    <col style={{ width: '60px' }} />
                </colgroup>
                <thead>
                    <tr>
                        <th>번호</th>
                        <th>제목</th>
                        <th>처리 기관</th>
                        <th>신청일</th>
                        <th>추진상황</th>
                        <th>조회</th>
                    </tr>
                </thead>
                <tbody>
                    {data.map((item) => (
                        <tr key={item.id}>
                            <td>{item.id}</td>
                            <td className="title">
                                <a href="#">{item.title}</a>
                                {item.badge && <Badge>{item.badge}</Badge>}
                            </td>
                            <td>{item.dept}</td>
                            <td>{item.date}</td>
                            <td>{item.status}</td>
                            <td>{item.views}</td>
                        </tr>
                    ))}
                </tbody>
            </Table>
        </TableWrapper>
    );
};

export default DataTable;
