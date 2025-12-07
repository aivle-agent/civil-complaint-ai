import React, { useState } from 'react';
import styled from 'styled-components';
import GlobalStyle from './styles/GlobalStyle';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import Footer from './components/Layout/Footer';
import ProposalList from './pages/ProposalList';
import ProposalForm from './components/Proposal/ProposalForm';

import InstitutionSelect from './components/Proposal/InstitutionSelect';

const MainContainer = styled.div`
  max-width: 1200px;
  margin: 40px auto;
  padding: 0 24px;
  display: flex;
  min-height: 600px;
`;

const ContentWrapper = styled.main`
  flex: 1;
`;

function App() {
    const [activePage, setActivePage] = useState('form'); // Default to form as requested
    const [proposalData, setProposalData] = useState({
        prplTitl: '',
        prplCntnCl: '',
        btmtIdeaCl: '',
        expcEfctCl: ''
    });
    // LangGraph 결과를 저장하여 ProposalForm과 InstitutionSelect 간 공유
    const [aiResult, setAiResult] = useState(null);

    const handleProposalDataChange = (name, value) => {
        setProposalData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    return (
        <>
            <GlobalStyle />
            <Header />
            <MainContainer>
                <Sidebar activePage={activePage} onNavigate={setActivePage} />
                <ContentWrapper>
                    {activePage === 'list' && <ProposalList />}
                    {activePage === 'form' && (
                        <ProposalForm
                            onNext={() => setActivePage('institution')}
                            data={proposalData}
                            onDataChange={handleProposalDataChange}
                            aiResult={aiResult}
                            onAiResult={setAiResult}
                        />
                    )}
                    {activePage === 'institution' && (
                        <InstitutionSelect
                            onNavigate={setActivePage}
                            data={proposalData}
                            aiResult={aiResult}
                        />
                    )}
                </ContentWrapper>
            </MainContainer>
            <Footer />
        </>
    );
}

export default App;

