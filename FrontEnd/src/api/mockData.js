import { faker } from '@faker-js/faker/locale/en';
import dayjs from 'dayjs';

const randomDate = (daysAgo) => faker.date.recent({ days: daysAgo });

// --- MOCK POSITIONS ---
export const generateMockPositions = (count = 10) => {
  const positions = [];
  for (let i = 1; i <= count; i++) {
    const openedAt = randomDate(30);
    const isClosed = faker.datatype.boolean({ probability: 0.3 });
    const score = faker.number.int({ min: 40, max: 95 });
    
    // Simulate cv analysis for candidate view
    const hasAnalysis = faker.datatype.boolean({ probability: 0.7 });
    let overallStatus = 'PENDING';
    if (score >= 90) overallStatus = 'EXCELLENT_MATCH';
    else if (score >= 70) overallStatus = 'GOOD_MATCH';
    else if (score >= 60) overallStatus = 'POTENTIAL';
    else overallStatus = 'POOR_FIT';

    positions.push({
      id: i,
      hrId: 'hr-admin',
      name: faker.person.jobTitle(), // e.g. "Intern Java Developer"
      language: faker.helpers.arrayElement(['Java', 'Python', 'React', 'NodeJS', 'C#', 'Go']),
      level: faker.helpers.arrayElement(['Intern', 'Junior', 'Middle', 'Senior', 'Lead']),
      isActive: !isClosed,
      openedAt: openedAt.toISOString(),
      closedAt: isClosed ? dayjs(openedAt).add(faker.number.int({ min: 5, max: 20 }), 'day').toISOString() : null,
      minFitScore: faker.number.int({ min: 60, max: 80 }),
      internalCount: faker.number.int({ min: 0, max: 50 }),
      externalCount: faker.number.int({ min: 0, max: 100 }),
      jobDescription: faker.lorem.paragraphs(2),
      status: 'PROCESSED',
      // For candidate view:
      cvAnalysis: hasAnalysis ? {
        score: score,
        overallStatus: overallStatus,
        isApplied: faker.datatype.boolean({ probability: 0.3 }),
        learningPath: score < 70 ? faker.lorem.paragraphs(1) : null
      } : null
    });
  }
  return positions;
};

// --- MOCK CANDIDATES ---
export const generateMockCandidates = (count = 30) => {
  const candidates = [];
  const stages = ['APPLIED', 'INTERVIEW_SCHEDULED', 'INTERVIEWED', 'OFFER', 'ACCEPTED', 'REJECTED'];
  
  for (let i = 1; i <= count; i++) {
    const isMaster = faker.datatype.boolean({ probability: 0.1 });
    const hasAnalysis = faker.datatype.boolean({ probability: 0.9 });
    const technicalScore = faker.number.int({ min: 40, max: 95 });
    const experienceScore = faker.number.int({ min: 40, max: 95 });
    const avgScore = Math.floor((technicalScore + experienceScore) / 2);
    const stage = faker.helpers.arrayElement(stages);
    const type = faker.helpers.arrayElement(['INTERNAL', 'EXTERNAL']);
    
    // Fake interview date if scheduled or later
    let interviewDate = null;
    if (['INTERVIEW_SCHEDULED', 'INTERVIEWED', 'OFFER', 'ACCEPTED'].includes(stage)) {
      interviewDate = dayjs().add(faker.number.int({ min: -5, max: 5 }), 'day').toISOString();
    }

    candidates.push({
      id: i,
      candidateId: `cand-${faker.string.uuid()}`,
      position_id: isMaster ? null : faker.number.int({ min: 1, max: 10 }),
      parentCvId: isMaster ? null : faker.number.int({ min: 100, max: 200 }),
      type: type, // INTERNAL (HR Upload) or EXTERNAL (Candidate Upload)
      name: faker.person.fullName(),
      email: faker.internet.email(),
      cvStatus: 'PARSED',
      recruitmentStage: stage,
      updatedAt: randomDate(10).toISOString(),
      interviewDate: interviewDate,
      driveFileUrl: 'https://docs.google.com/document/d/1_dummy_link/view',
      analysis: hasAnalysis ? {
        positionName: faker.person.jobTitle(), // Lấy tạm theo mock
        technicalScore,
        experienceScore,
        overallScore: avgScore,
        overallStatus: avgScore >= 70 ? 'MATCHED' : 'NOT_MATCHED',
        skillMatch: faker.lorem.sentences(1),
        skillMiss: faker.lorem.sentences(1),
        learningPath: avgScore < 70 ? faker.lorem.paragraphs(1) : null,
        feedback: faker.lorem.sentence() // Lý do fit/not fit (Reason for match)
      } : null
    });
  }
  return candidates;
};

// Singleton data
export const mockPositionsData = generateMockPositions(10);
export const mockCandidatesData = generateMockCandidates(30);

// --- API MOCKS ---

export const fetchPositions = async () => {
  return new Promise(resolve => setTimeout(() => resolve([...mockPositionsData]), 800));
};

export const fetchActivePositions = async () => {
  return new Promise(resolve => setTimeout(() => resolve(mockPositionsData.filter(p => p.isActive)), 500));
};

export const updatePositionScore = async (id, newScore) => {
  return new Promise(resolve => {
    setTimeout(() => {
      const pos = mockPositionsData.find(p => p.id === id);
      if (pos) pos.minFitScore = newScore;
      resolve(pos);
    }, 500);
  });
};

export const togglePositionActive = async (id, isActive) => {
  return new Promise(resolve => {
    setTimeout(() => {
      const pos = mockPositionsData.find(p => p.id === id);
      if (pos) {
        pos.isActive = isActive;
        if (!isActive) pos.closedAt = dayjs().toISOString();
        else pos.closedAt = null;
      }
      resolve(pos);
    }, 500);
  });
};

export const fetchCandidates = async () => {
  return new Promise(resolve => setTimeout(() => resolve([...mockCandidatesData]), 800));
};

export const updateCandidateStage = async (id, newStage) => {
  return new Promise(resolve => {
    setTimeout(() => {
      const cand = mockCandidatesData.find(c => c.id === id);
      if (cand) cand.recruitmentStage = newStage;
      resolve(cand);
    }, 500);
  });
};

export const scheduleCandidateInterview = async (id, data) => {
  return new Promise(resolve => {
    setTimeout(() => {
      const cand = mockCandidatesData.find(c => c.id === id);
      if (cand) {
        cand.recruitmentStage = 'INTERVIEW_SCHEDULED';
        cand.interviewDate = data.date;
      }
      resolve(cand);
    }, 500);
  });
};

export const updateCandidateCVInfo = async (id, data) => {
  return new Promise(resolve => {
    setTimeout(() => {
      const cand = mockCandidatesData.find(c => c.id === id);
      if (cand) {
        cand.name = data.name;
        cand.email = data.email;
      }
      resolve(cand);
    }, 500);
  });
};
